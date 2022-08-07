import json
import os
import sys
import shutil

import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

from CCdefault.app_entity.config_entity import DataValidationConfig
from CCdefault.app_entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact
from CCdefault.app_logger import App_Logger
from CCdefault.app_database.mongoDB import MongoDB
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_util.util import read_yaml_file , get_last_experiment_data , compare_two_csv
from CCdefault.app_constants import DATA_VALIDATION_SCHEMA_KEY

class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.logger = App_Logger("Data_validation")
            self.logger.info(f"{'>>' * 30}Data Validation log started.{'<<' * 30} \n\n")
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_train_and_test_df(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df, test_df
        except Exception as e:
            raise App_Exception(e, sys) from e

    def is_train_test_file_exists(self) -> bool:
        try:
            self.logger.info("Checking if training and test file is available")
            is_train_file_exist = False
            is_test_file_exist = False

            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            is_train_file_exist = os.path.exists(train_file_path)
            is_test_file_exist = os.path.exists(test_file_path)

            is_available = is_train_file_exist and is_test_file_exist

            self.logger.info(f"Is train and test file exists?-> {is_available}")

            if not is_available:
                training_file = self.data_ingestion_artifact.train_file_path
                testing_file = self.data_ingestion_artifact.test_file_path
                message = f"Training file: {training_file} or Testing file: {testing_file}" \
                          "is not present"
                raise Exception(message)

            return is_available
        except Exception as e:
            raise App_Exception(e, sys) from e

    def validate_dataset_schema(self, ) -> bool:
        try:
            self.logger.info("Validating dataset schema")
            validation_status = False
            schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
            schema_dict = schema_config[DATA_VALIDATION_SCHEMA_KEY]
            train_df, test_df = self.get_train_and_test_df()

            for column, data_type in schema_dict.items():
                train_df[column].astype(data_type)
                test_df[column].astype(data_type)
            self.logger.info("Dataset schema validation completed")
            validation_status = True
            self.logger.info(f"Validation_status {validation_status}")
            return validation_status
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_and_save_data_drift_report(self):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df, previous_df = self.get_train_and_previous_df()

            if previous_df is None:
                self.logger.info("Previous dataframes is not available")
                report = False
            else:

                profile.calculate(train_df, previous_df)
                dashboard = Dashboard(tabs=[DataDriftTab()])
                report = json.loads(profile.json())
                report_page_file_path = self.data_validation_config.report_page_file_path
                report_file_path = self.data_validation_config.report_file_path
                report_dir = os.path.dirname(report_file_path)
                os.makedirs(report_dir, exist_ok=True)
                dashboard.save(report_page_file_path)

                with open(report_file_path, "w") as report_file:
                    json.dump(report, report_file, indent=6)
            return report
        except Exception as e:
            raise App_Exception(e, sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            status = False
            report = self.get_and_save_data_drift_report()       
            if report :
                status = report['data_drift'] [ 'data' ]["metrics" ]['dataset_drift']
            return status
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    def get_experiment_status(self, train_file_path ,test_file_path,validated_test_file_path, validated_train_file_path):
            
        last_experiment_test_file_path = get_last_experiment_data(validated_test_file_path)
        last_experiment_train_file_path = get_last_experiment_data(validated_train_file_path)
        if os.path.exists(last_experiment_train_file_path) :
            train_diff = compare_two_csv(current_path=train_file_path, previous_path=last_experiment_train_file_path ,
                                        key_columns="default.payment.next.month")
            test_diff = compare_two_csv(current_path=test_file_path, previous_path=last_experiment_test_file_path,
                                        key_columns="default.payment.next.month")
            self.logger.info(f"Train diff: {train_diff}")
            if train_diff == "same":
                self.logger.info("Train file is same content as previous experiment file")
                return True
        else:
            self.logger.info("No previous experiment file found")
            return False
    

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            is_validated = True
            message="Data Validation performed successfully."
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            data_validated_artifact_dir = self.data_validation_config.data_validated_artifact_dir
            validated_test_file_path = os.path.join(data_validated_artifact_dir,os.path.basename(test_file_path))
            validated_train_file_path = os.path.join(data_validated_artifact_dir,os.path.basename(train_file_path))
            last_experiment_status = self.get_experiment_status( train_file_path ,test_file_path,validated_test_file_path, validated_train_file_path)
            if last_experiment_status:
                message = "Data Validation already performed successfully"
                self.logger.info(message)
            
            else:
                test_collections = self.data_validation_config.data_validated_test_collection
                train_collections = self.data_validation_config.data_validated_train_collection
                test_connection = MongoDB(test_collections, drop_collection=True)
                train_connection = MongoDB(train_collections, drop_collection=True)
                
                test_train_file_status = self.is_train_test_file_exists()
                validation_status = self.validate_dataset_schema()
                data_drift= self.is_data_drift_found()
                if data_drift:
                    raise Exception("Data drift found")
                if test_train_file_status and validation_status:
                    self.logger.info("Data validation completed")
                    os.makedirs(os.path.dirname(validated_train_file_path) , exist_ok=True)
                    shutil.copyfile(test_file_path, validated_test_file_path)
                    shutil.copyfile(train_file_path, validated_train_file_path)
                    test_df = pd.read_csv(validated_test_file_path)
                    train_df= pd.read_csv(validated_train_file_path)
                    test_connection.Insert_Many(test_df.to_dict(orient='records'))
                    train_connection.Insert_Many(train_df.to_dict(orient='records'))
                else:
                    message="Data Validation not successfully."
                    validation_status = False
                    self.logger.info("Data Validation not successfully.")
                    
            data_validation_artifact = DataValidationArtifact(
                    schema_file_path=self.data_validation_config.schema_file_path,
                    report_file_path=self.data_validation_config.report_file_path,
                    report_page_file_path=self.data_validation_config.report_page_file_path,
                    is_validated=is_validated,
                    message=message)
            return data_validation_artifact
        
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_train_and_previous_df(self) -> pd.DataFrame:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            train_df = pd.read_csv(train_file_path)
            previous_file_path = get_last_experiment_data(train_file_path)
            
            if os.path.samefile(train_file_path, previous_file_path):
                previous_df = pd.read_csv(train_file_path)
            else:
                previous_df = pd.read_csv(previous_file_path)
            return train_df, previous_df
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        self.logger.info(f"{'>>' * 30}Data Validation log completed.{'<<' * 30} \n\n")