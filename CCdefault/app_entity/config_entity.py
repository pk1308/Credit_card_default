from collections import namedtuple

DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["dataset_download_url" , 
                                  "dataset_download_file_name", "raw_data_file_path",
                                  "raw_file_path_to_ingest" , "ingested_train_file_path", 
                                  "ingested_test_data_path", "ingested_train_collection", 
                                  "ingested_test_collection"])

DataValidationConfig = namedtuple("DataValidationConfig",
                                  ['data_validated_artifact_dir',"schema_file_path", "report_file_path",
                                   "report_page_file_path" , "data_validated_test_collection" , "data_validated_train_collection"])

DataTransformationConfig = namedtuple("DataTransformationConfig", ["transformed_train_dir",
                                                                   "transformed_test_dir",
                                                                   "preprocessed_object_file_path" , 
                                                                   "ingested_train_collection" , 
                                                                   'ingested_test_collection'])

ModelTrainerConfig = namedtuple("ModelTrainerConfig",["model_config_file_path",
                                                       "base_accuracy","trained_model_file_path", 'model_report_dir'])

ModelEvaluationConfig = namedtuple("ModelEvaluationConfig", ["model_evaluation_file_path","time_stamp"])

ModelPusherConfig = namedtuple("ModelPusherConfig", ["export_dir_path"])


TrainingPipelineConfig = namedtuple("TrainingPipelineConfig",["artifact_dir", "pipeline_name"])