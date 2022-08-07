
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact
from CCdefault.app_entity.config_entity import ModelTrainerConfig
from CCdefault.app_entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel, \
    evaluate_classification_model
from CCdefault.app_util.util import load_numpy_array_data, save_object, load_object
import os
import sys
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

logging = App_Logger("Model_trainer")

class EstimatorModel:
    """model estimator : Train the model and save the model to pickle """
    def __init__(self, preprocessing_object, trained_model_dict):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_dict:  {cluster : modelsavepath}
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_dict = trained_model_dict
        self.trained_model_object = {cluster : load_object(model) for cluster , model in trained_model_dict.items()}

    def single_predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        assert len(X) == 1, "single_predict function only accepts single input"
        transformed_feature = self.preprocessing_object.transform(X)
        cluster = transformed_feature["cluster"]
        prediction = self.trained_model_object[cluster].predict(transformed_feature.drop("cluster", axis=1))
        return prediction
    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        prediction = np.ones(len(transformed_feature))
        for row_number in range(len(transformed_feature)):
            to_predict = transformed_feature[row_number,:-1].reshape(1, -1)
            cluster = transformed_feature[row_number,-1]
            model = self.trained_model_object[cluster]
            prediction[row_number] = model.predict(to_predict)
        return prediction

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_arr = load_numpy_array_data(transformed_train_file_path)
            
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            model_report_dir = self.model_trainer_config.model_report_dir
            cluster_model_dict = dict()
            for cluster in np.unique(train_arr[ : , -2]):
                logging.info(f"Training model for cluster {cluster}")
                to_train_arr = train_arr[train_arr[ : , -2] == cluster]
                to_train_features = to_train_arr[:, :-2]
                to_train_labels = to_train_arr[:, -1]

                X_train, X_test, y_train, y_test = train_test_split(to_train_features, to_train_labels, test_size=0.2, random_state=1965)
                logging.info("Extracting model config file path")
                model_config_file_path = self.model_trainer_config.model_config_file_path

                logging.info(f"Initializing model factory class using above model config file: {model_config_file_path}")
                model_factory = ModelFactory(model_config_path=model_config_file_path)

                base_accuracy = self.model_trainer_config.base_accuracy
                logging.info(f"Expected accuracy: {base_accuracy}")

                logging.info("Initiating operation model selection")
                best_model = model_factory.get_best_model(X=X_train, y=y_train, base_accuracy=base_accuracy)

                logging.info(f"Best model found on training dataset: {best_model}")

                logging.info("Extracting trained model list.")
                grid_searched_best_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

                model_list = [model.best_model for model in grid_searched_best_model_list]
                logging.info(f"Model list: {model_list} , {len(model_list)}")
                cluster_name = f"cluster_{cluster}"
                report_dir = os.path.join(model_report_dir, cluster_name)
                logging.info("Evaluation all trained model on training and testing dataset both")
                metric_info: MetricInfoArtifact = evaluate_classification_model(estimators=model_list, X_train=X_train,
                                                                            y_train=y_train, X_test=X_test, y_test=y_test,
                                                                            base_accuracy=base_accuracy , report_dir=report_dir,
                                                                            is_fitted=True)
                logging.info(f"Metric info: {metric_info}")
        
                model_file_name = f"{metric_info.model_name}.pkl"
                model_file_path = os.path.join(report_dir, cluster_name, model_file_name)
                model_object = metric_info.model_object
                cluster_model_dict[cluster] = model_file_path
                save_object(file_path=model_file_path, obj=model_object)
                
            logging.info("Best found model on both training and testing dataset.")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            
            model_dict = {cluster: model_file_path for cluster, model_file_path in cluster_model_dict.items()}

            prediction_model = EstimatorModel(preprocessing_object=preprocessing_obj, trained_model_dict=model_dict)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=prediction_model)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Trained successfully",
                                                          trained_model_file_path=trained_model_file_path,
                                                          train_f1="None",
                                                          test_f1="metric_info.test_f1",
                                                          train_precision="metric_info.train_precision",
                                                          test_precision="metric_info.test_precision",
                                                          model_accuracy="metric_info.model_accuracy")

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
