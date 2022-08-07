

import sys
import  os
import  json

from CCdefault.app_entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig,\
DataTransformationConfig, ModelTrainerConfig,ModelEvaluationConfig , ModelPusherConfig
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_util.util import read_yaml_file
from CCdefault.app_constants import *

logging = App_Logger(__name__)


class Configuration:

    def __init__(self,
                 config_file_path: str = CONFIG_FILE_PATH) -> None:
        try:
            self.config_info = read_yaml_file(file_path=config_file_path)
            self.pipeline_config = self.get_training_pipeline_config()
            self.time_stamp = CURRENT_TIME_STAMP

        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            logging.info(f"{'>>' * 20}Data Ingestion config.{'<<' * 20} \n\n")
            artifact_dir = self.pipeline_config.artifact_dir
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_download_url = data_ingestion_config_info[DATA_INGESTION_DATA_KEY]
            data_download_file_name = data_ingestion_config_info[DATA_INGESTION_DOWNLOAD_FILE_NAME_KEY]
            data_ingestion_dir_name = data_ingestion_config_info[DATA_INGESTION_DIR_KEY]
            raw_data_dir_name = data_ingestion_config_info[DATA_INGESTION_RAW_DIR_KEY]
            raw_data_file_name = data_ingestion_config_info[DATA_INGESTION_RAW_DATA_FILE_NAME_KEY]
            ingested_data_dir_name = data_ingestion_config_info[DATA_INGESTION_INGESTED_DIR_KEY]
            ingested_train_filename = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY]
            ingested_test_filename = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY]
            ingested_train_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY]
            ingested_test_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY]
            data_ingestion_time_stamp = self.time_stamp
            data_ingestion_dir = os.path.join(artifact_dir, data_ingestion_dir_name )
            raw_data_dir = os.path.join(data_ingestion_dir, raw_data_dir_name , data_ingestion_time_stamp)
            raw_data_file_path = os.path.join(raw_data_dir, data_download_file_name)
            raw_data_file_path_to_ingest = os.path.join(raw_data_dir,raw_data_file_name)
            data_ingested_dir = os.path.join(data_ingestion_dir, ingested_data_dir_name , data_ingestion_time_stamp)
            ingested_train_file_path = os.path.join(data_ingested_dir,ingested_train_filename)
            ingested_test_file_path = os.path.join(data_ingested_dir, ingested_test_filename)  
            


            data_ingestion_config = DataIngestionConfig(dataset_download_url=data_download_url,
                                                        dataset_download_file_name = data_download_file_name,
                                                        raw_data_file_path=raw_data_file_path,
                                                        raw_file_path_to_ingest=raw_data_file_path_to_ingest,
                                                        ingested_train_file_path=ingested_train_file_path,
                                                        ingested_test_data_path=ingested_test_file_path,
                                                        ingested_train_collection=ingested_train_collection,
                                                        ingested_test_collection=ingested_test_collection)
            logging.info(f"Data ingestion config: {data_ingestion_config}")
            return data_ingestion_config
            
            
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_training_pipeline_config(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config = self.config_info[TRAINING_PIPELINE_CONFIG_KEY]
            pipeline_dir = training_pipeline_config[TRAINING_PIPELINE_NAME_KEY]
            pipeline_artifact_dir = training_pipeline_config[TRAINING_PIPELINE_ARTIFACT_DIR_KEY]
            artifact_dir = os.path.join(ROOT_DIR,pipeline_dir , pipeline_artifact_dir)
            os.makedirs(artifact_dir, exist_ok=True)
            training_pipeline_config = TrainingPipelineConfig(artifact_dir=artifact_dir,
                                                              pipeline_name=pipeline_dir)
            logging.info(f"Training pipeline config: {training_pipeline_config}")
            return training_pipeline_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir

            data_validation_config_info = self.config_info[DATA_VALIDATION_CONFIG_KEY]
            data_validation_dir = data_validation_config_info[DATA_VALIDATION_DIR_KEY]
            schema_dir = data_validation_config_info[DATA_VALIDATION_SCHEMA_DIR_KEY]
            schema_file_name = data_validation_config_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY]
            report_page_file_name = data_validation_config_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            report_dir = data_validation_config_info[DATA_VALIDATION_REPORT_DIR_KEY]
            report_file_name = data_validation_config_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY]
            data_validated_test_collection = data_validation_config_info[DATA_VALIDATED_TEST_COLLECTION_KEY]
            data_validated_train_collection = data_validation_config_info[DATA_VALIDATED_TRAIN_COLLECTION_KEY]
            schema_file_path = os.path.join(ROOT_DIR,schema_dir, schema_file_name)
            data_validation_artifact_dir = os.path.join(artifact_dir, data_validation_dir,self.time_stamp)
            report_file_path = os.path.join(data_validation_artifact_dir,report_dir, report_file_name)
            report_page_file_path = os.path.join(data_validation_artifact_dir,report_dir, report_page_file_name)
            data_validated_artifact_dir = os.path.join(data_validation_artifact_dir,"data_validated")

            data_validation_config = DataValidationConfig(data_validated_artifact_dir= data_validated_artifact_dir,
                                                        schema_file_path=schema_file_path,
                                                          report_file_path=report_file_path,
                                                          report_page_file_path=report_page_file_path,
                                                          data_validated_test_collection=data_validated_test_collection,
                                                          data_validated_train_collection=data_validated_train_collection)
            return data_validation_config
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            data_transformation_config_info = self.config_info[DATA_TRANSFORMATION_CONFIG_KEY]
            data_ingestion_config_info = self.config_info[DATA_INGESTION_CONFIG_KEY]
            data_transformation_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_DIR_KEY]
            transformed_dir = data_transformation_config_info[DATA_TRANSFORMATION_DIR_NAME_KEY]
            transformed_train_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY]
            transformed_test_dir_name = data_transformation_config_info[DATA_TRANSFORMATION_TEST_DIR_NAME_KEY]
            preprocessing_dir = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY]
            preprocessed_obj_name = data_transformation_config_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            ingested_train_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY]
            ingested_test_collection = data_ingestion_config_info[DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY]
            data_transformation_dir= os.path.join(artifact_dir, data_transformation_dir_name,self.time_stamp)
            transformed_train_dir = os.path.join(data_transformation_dir, transformed_dir, transformed_train_dir_name)
            transformed_test_dir = os.path.join(data_transformation_dir, transformed_dir, transformed_test_dir_name)
            preprocessed_obj_path = os.path.join(data_transformation_dir, preprocessing_dir, preprocessed_obj_name)
            
            os.makedirs(os.path.dirname(transformed_train_dir), exist_ok=True)
            os.makedirs(os.path.dirname(transformed_test_dir), exist_ok=True)

            data_transformation_config = DataTransformationConfig(transformed_train_dir = transformed_train_dir,
                                                                   transformed_test_dir = transformed_test_dir,
                                                                   preprocessed_object_file_path = preprocessed_obj_path,
                                                                   ingested_train_collection= ingested_train_collection,
                                                                   ingested_test_collection= ingested_test_collection)
            return data_transformation_config
            
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        
        try:
            artifact_dir = self.pipeline_config.artifact_dir
            model_trainer_config_info = self.config_info[MODEL_TRAINER_CONFIG_KEY]
            
            model_trainer_dir = model_trainer_config_info[MODEL_TRAINER_ARTIFACT_DIR]
            model_config_dir = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_DIR_KEY]
            model_config_file_name = model_trainer_config_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY]
            base_accuracy = model_trainer_config_info[MODEL_TRAINER_BASE_ACCURACY_KEY]
            trained_model_dir_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_DIR_KEY]
            trained_model_file_name = model_trainer_config_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY]
            
            model_report_dir_name = model_trainer_config_info[MODEL_TRAINER_MODEL_REPORT_DIR_KEY]
            model_config_file_path = os.path.join(model_config_dir, model_config_file_name)
            model_trainer_artifact_dir = os.path.join(artifact_dir, model_trainer_dir)
            model_report_dir = os.path.join(model_trainer_artifact_dir, model_report_dir_name , self.time_stamp)
            trained_model_file_path = os.path.join(model_trainer_artifact_dir, trained_model_dir_name,self.time_stamp,
                                                   trained_model_file_name)
            
            
        

            model_config_file_path = os.path.join(model_config_dir, model_config_file_name)
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)


            model_trainer_config = ModelTrainerConfig( model_config_file_path= model_config_file_path,
                                                      base_accuracy= base_accuracy,
                                                      trained_model_file_path= trained_model_file_path,
                                                      model_report_dir= model_report_dir)
    
            logging.info(f"Model trainer config: {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        try:
            model_evaluation_config = self.config_info[MODEL_EVALUATION_CONFIG_KEY]
            artifact_dir = self.pipeline_config.artifact_dir
            model_evaluation_dir_name = model_evaluation_config[MODEL_EVALUATION_DIR_KEY]
            model_evaluation_file_name = model_evaluation_config[MODEL_EVALUATION_FILE_NAME_KEY]
            model_evaluation_file_path = os.path.join(artifact_dir,model_evaluation_dir_name, model_evaluation_file_name)
            response = ModelEvaluationConfig(model_evaluation_file_path=model_evaluation_file_path,
                                            time_stamp=self.time_stamp)
            logging.info(f"Model Evaluation Config: {response}.")
            return response
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_model_pusher_config(self) -> ModelPusherConfig:
        try:
            time_stamp = f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
            model_pusher_config_info = self.config_info[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(ROOT_DIR, model_pusher_config_info[MODEL_PUSHER_MODEL_EXPORT_DIR_KEY],
                                           time_stamp)

            model_pusher_config = ModelPusherConfig(export_dir_path=export_dir_path)
            logging.info(f"Model pusher config {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise App_Exception(e, sys) from e