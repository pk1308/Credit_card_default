
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact , DataIngestionArtifact
from CCdefault.app_entity.config_entity import ModelTrainerConfig
from CCdefault.app_entity.model_factory import MetricInfoArtifact, ModelFactory, GridSearchedBestModel, \
    evaluate_classification_model
import pandas as pd 
from CCdefault.app_util.util import load_numpy_array_data, save_object, load_object , reduce_mem_usage
import os
import sys
from typing import List
import numpy as np
from sklearn.model_selection import train_test_split

logging = App_Logger("Model_trainer")

class BaseModel:
    """model estimator : Train the model and save the model to pickle """
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_dict:  {cluster : model saved path}
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object
       
    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it perform prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        prediction = self.trained_model_object.predict(transformed_feature[:, :-1])
        return prediction

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"

class EstimatorModel:
    """model estimator : Train the model and save the model to pickle """
    def __init__(self, preprocessing_object, trained_model_dict):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_dict:  {cluster : model saved path}
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_dict = trained_model_dict
        self.trained_model_object = {cluster : load_object(model) for cluster , model in trained_model_dict.items()}

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
                 data_transformation_artifact: DataTransformationArtifact , data_ingestion_artifact:  DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            train_arr = load_numpy_array_data(transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            logging.info(f"{'>>' * 30}Base Model.{'<<' * 30} ")
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            model_report_dir = self.model_trainer_config.model_report_dir
            cluster_model_dict = dict()
            base_X = train_arr[:,:-2]
            base_y = train_arr[:,-1]
            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected accuracy: {base_accuracy}")
            logging.info("Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path
            base_x_train , base_x_test , base_y_train , base_y_test = train_test_split(base_X, base_y, test_size=0.2, random_state=1965)
            base_model_factory = ModelFactory(model_config_path=model_config_file_path)
            base_best_model =base_model_factory.get_best_model(X=base_x_train, y=base_y_train, base_accuracy=base_accuracy)
            base_grid_searched_best_model_list: List[GridSearchedBestModel] = base_model_factory.grid_searched_best_model_list
            base_model_list = [model.best_model for model in base_grid_searched_best_model_list]
            base_report_dir = os.path.join(model_report_dir, "Base_model")
            base_metric_info: MetricInfoArtifact = evaluate_classification_model(estimators=base_model_list, X_train=base_x_train,
                                                                            y_train=base_y_train, X_test=base_x_test, y_test=base_y_test,
                                                                            base_accuracy=base_accuracy , report_dir=base_report_dir,
                                                                            is_fitted=True)
            logging.info(f"{base_metric_info.__dict__}")
            base_model_file_name = f"{base_metric_info.model_name}.pkl"
            base_model_file_path = os.path.join(base_report_dir, "Model", base_model_file_name)
            base_model_object = base_metric_info.model_object
            base_predictor = BaseModel(preprocessing_object=preprocessing_obj , trained_model_object=base_model_object)
            save_object(file_path=base_model_file_path, obj=base_predictor)
            
            logging.info(f"{'>>' * 30}Base model done{'<<' * 30} ")
            
            for cluster in np.unique(train_arr[ : , -2]):
                logging.info(f"{'>>' * 30}Training model for cluster {cluster}{'>>' * 30}")
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
                model_file_path = os.path.join(report_dir, "Model", model_file_name)
                model_object = metric_info.model_object
                cluster_model_dict[cluster] = model_file_path
                save_object(file_path=model_file_path, obj=model_object)
                
            logging.info("Best found model on both training and testing dataset.")

            
            model_dict = {cluster: model_file_path for cluster, model_file_path in cluster_model_dict.items()}

            prediction_model = EstimatorModel(preprocessing_object=preprocessing_obj, trained_model_dict=model_dict)
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            train_df = reduce_mem_usage(train_df)
            test_df = reduce_mem_usage(test_df)
            X_train = train_df.drop(columns=['default.payment.next.month', "ID"], axis=1)
            y_train = train_df['default.payment.next.month']
            X_test = test_df.drop(columns=['default.payment.next.month', "ID"], axis=1)
            y_test = test_df['default.payment.next.month']
            cluster_report_dir = os.path.join(model_report_dir, "cluster_custom_model")
            base_accuracy = self.model_trainer_config.base_accuracy
            clustered_model_list = [prediction_model , base_predictor]
            metric_info = evaluate_classification_model(estimators=clustered_model_list, X_train=X_train,
                                                                            y_train=y_train, X_test=X_test, y_test=y_test,base_accuracy=base_accuracy , 
                                                                            report_dir=cluster_report_dir,
                                                                            is_fitted=True)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=prediction_model)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True, message="Model Trained successfully",
                                                          trained_model_file_path=trained_model_file_path,
                                                          train_f1=metric_info.test_f1,
                                                          test_f1=metric_info.test_f1,
                                                          train_precision=metric_info.train_precision,
                                                          test_precision=metric_info.test_precision,
                                                          model_accuracy=metric_info.model_accuracy)

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            logging.error(e)
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
