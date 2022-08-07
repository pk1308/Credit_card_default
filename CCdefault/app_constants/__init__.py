import os
from datetime import datetime


def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}"


ROOT_DIR = os.getcwd()  # to get current working directory
CURRENT_TIME_STAMP = get_current_time_stamp()


# config constants
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILE_NAME)

# Training pipeline related variable
TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline_config"
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = "artifact_dir"
TRAINING_PIPELINE_NAME_KEY = "pipeline_name"

# Data Ingestion related variable
DATA_INGESTION_CONFIG_KEY = "data_ingestion_config"
DATA_INGESTION_DIR_KEY = "data_ingestion_dir"
DATA_INGESTION_DATA_KEY = 'dataset_download_url'
DATA_INGESTION_DOWNLOAD_FILE_NAME_KEY = 'dataset_download_file_name'
DATA_INGESTION_INGESTED_DIR_KEY = "ingested_dir"
DATA_INGESTION_RAW_DIR_KEY = "raw_data_dir"
DATA_INGESTION_RAW_DATA_FILE_NAME_KEY = "raw_data_file_name"
DATA_INGESTION_INGESTED_TRAIN_FILE_NAME_KEY = "ingested_data_Train_file_name"
DATA_INGESTION_INGESTED_TEST_FILE_NAME_KEY = "ingested_data_Test_file_name"
DATA_INGESTION_INGESTED_TRAIN_COLLECTION_KEY = "ingested_data_Train_collection_name"
DATA_INGESTION_INGESTED_TEST_COLLECTION_KEY ="ingested_data_Test_collection_name" 

# Data Validation related variables
DATA_VALIDATION_CONFIG_KEY = "data_validation_config"
DATA_VALIDATION_DIR_KEY = 'data_validation_dir'
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_ARTIFACT_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_DIR_KEY = "report_dir"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = "report_file_name"
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = "report_page_file_name"
DATA_VALIDATION_SCHEMA_KEY = "columns"
TARGET_COLUMN_KEY = "target_column"
COLUMNS_TO_CLUSTER_KEY = "columns_to_cluster"
DATA_VALIDATED_TEST_COLLECTION_KEY = "data_validated_test_collection_name"
DATA_VALIDATED_TRAIN_COLLECTION_KEY = "data_validated_train_collection_name"

# Data Transformation related variables
DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation_config"
DATA_TRANSFORMATION_DIR_KEY = "data_transformation_dir"
DATA_TRANSFORMATION_DIR_NAME_KEY = "transformed_dir"
DATA_TRANSFORMATION_TRAIN_DIR_NAME_KEY = "transformed_train_dir"
DATA_TRANSFORMATION_TEST_DIR_NAME_KEY = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSING_DIR_KEY = "preprocessing_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_object_file_name"


# Model Training related variables

MODEL_TRAINER_ARTIFACT_DIR = "model_trainer_dir"
MODEL_TRAINER_CONFIG_KEY = "model_trainer_config"
MODEL_TRAINER_TRAINED_MODEL_DIR_KEY = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME_KEY = "model_file_name"
MODEL_TRAINER_MODEL_REPORT_DIR_KEY = "model_report_dir"
MODEL_TRAINER_BASE_ACCURACY_KEY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR_KEY = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME_KEY = "model_config_file_name"


# Model Evaluation related variables
MODEL_EVALUATION_CONFIG_KEY = 'model_evaluation_config'
MODEL_EVALUATION_DIR_KEY = 'model_evaluation_dir'
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"
MODEL_EVALUATION_REPORT_DIR_NAME = "reports"


# Model Pusher config key
MODEL_PUSHER_CONFIG_KEY = "model_pusher_config"
MODEL_PUSHER_MODEL_EXPORT_DIR_KEY = "model_export_dir"

BEST_MODEL_KEY = "best_model"
HISTORY_KEY = "history"
MODEL_PATH_KEY = "model_path"

EXPERIMENT_DIR_NAME = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"