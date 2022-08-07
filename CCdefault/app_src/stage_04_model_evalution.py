import pandas as pd
import numpy as np
from tenacity import retry

from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_entity.config_entity import ModelEvaluationConfig
from CCdefault.app_entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact, \
    ModelTrainerArtifact, ModelEvaluationArtifact
from CCdefault.app_constants import *
from CCdefault.app_util.util import write_yaml_file, read_yaml_file, load_object
from CCdefault.app_entity.model_factory import evaluate_classification_model
import os
import sys

logging = App_Logger("model_Validation")


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            logging.info(f"{'>>' * 30}Model Evaluation log started.{'<<' * 30} ")
            self.model_evaluation_config = model_evaluation_config
    
            self.model_trainer_artifact = model_trainer_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise App_Exception(e, sys) from e

    def update_evaluation_report(self, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content

            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            write_yaml_file(file_path=eval_file_path, data=model_eval_content)

        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_object = load_object(file_path=trained_model_file_path)

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            schema_file_path = self.data_validation_artifact.schema_file_path
            schema_content = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_content[TARGET_COLUMN_KEY]
            selected_columns = schema_content[COLUMNS_TO_CLUSTER_KEY]
            train_dataframe = pd.read_csv(train_file_path , usecols=selected_columns)
                                                        
            test_dataframe = pd.read_csv(test_file_path , usecols=selected_columns)
            # target_column
            logging.info("Converting target column into numpy array.")
            y_train = np.array(train_dataframe[target_column_name])
            y_test = np.array(test_dataframe[target_column_name])
            logging.info("Conversion completed target column into numpy array.")

            # dropping target column from the dataframe
            logging.info("Dropping target column from the dataframe.")
            X_train= train_dataframe.drop(target_column_name, axis=1, inplace=True)
            X_test=test_dataframe.drop(target_column_name, axis=1, inplace=True)
            logging.info("Dropping target column from the dataframe completed.")

            model = self.get_best_model()

            if model is None:
                logging.info("Not found any existing model. Hence accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")
                return model_evaluation_artifact

            model_list = [model, trained_model_object]
            report_dir = self.model_evaluation_config.model_evaluation_report_dir

            metric_info_artifact = evaluate_classification_model(estimators=model_list, X_train=X_train,
                                                                            y_train=y_train, X_test=X_test,                                                                            y_test=y_test,
                                                                            base_accuracy=0.6 , 
                                                                            report_dir=report_dir,
                                                                            is_fitted=True)
            logging.info(f"Model evaluation completed. model metric artifact: {metric_info_artifact}")

            if metric_info_artifact is None:
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response

            if metric_info_artifact.model_index  == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Model accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is no better than existing model hence not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")