from cmath import log
import logging
from operator import index
import shutil
import sys, os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import kaggle
import subprocess
from zipfile import ZipFile
from CCdefault.app_config.configuration import Configuration
from CCdefault.app_entity.config_entity import DataIngestionConfig
from CCdefault.app_entity.artifacts_entity import DataIngestionArtifact
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_database.mongoDB import MongoDB
from CCdefault.app_util.util import reduce_mem_usage ,  get_last_experiment_data , write_yaml_file , compare_two_csv
from CCdefault.app_constants import CURRENT_TIME_STAMP




class DataIngestion:
    """Stage 1 data ingestion : Download dataset, split data into train and test, export to pickle and mongoDb
     Input :
     DataIngestionConfig = namedtuple("DataIngestionConfig",
                                 ["dataset_download_url" ,
                                  "dataset_download_file_name", "raw_data_file_path",
                                  "raw_file_path_to_ingest" , "ingested_train_file_path",
                                  "ingested_test_data_path", "ingested_train_collection",
                                  "ingested_test_collection"])

     output :
       DataIngestionArtifact(train_file_path,
                            test_file_path,
                            is_ingested : bool,
                            message=f"Data ingestion completed successfully.")
        top download the dataset from kaggle we use kaggle api authentication
        reference : https://github.com/Kaggle/kaggle-api for more details on kaggle api"""

    def __init__(self, data_ingestion_config_info: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config_info
            self.logger = App_Logger("Data_Ingestion")
            self.logger.info(f"{'>>' * 20}Experiment : PCA {'<<' * 20}")
        except Exception as e:
            raise App_Exception(e, sys)
        
    def download_data(self,dataset_download_url: str, dataset_download_file_name: str,
                      raw_data_file_path: str) -> str:
        """Download the dataset from kaggle unzip move to raw data folder

        Args:
            dataset_download_url (str): the data set key in kaggle 
            dataset_download_file_name (str): name of the dataset zip file
            raw_data_file_path (str): path to raw data folder with file name 

        Raises:
            App_Exception: _description_

        Returns:
            str: _description_
        """
        try:
            # extraction remote url to download dataset
            self.logger.info(f"Downloading dataset from kaggle")
            kaggle.api.authenticate() # authenticate to kaggle api
            # run  sys in the terminal
            subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_download_url])
            self.logger.info(f"Dataset downloaded successfully file name : {dataset_download_file_name}")
            os.makedirs(os.path.dirname(raw_data_file_path), exist_ok=True)
            shutil.move(dataset_download_file_name, raw_data_file_path)
            self.logger.info(f"Dataset moved to raw data folder : {raw_data_file_path}")
            with ZipFile(raw_data_file_path, 'r') as zipObj:

                zipObj.extractall(os.path.dirname(raw_data_file_path))
            self.logger.info("Dataset unzipped successfully")

            return True

        except Exception as e:
            raise App_Exception(e, sys) from e

    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            self.logger.info(f"{'>>' * 20}Data splitting.{'<<' * 20}")
            raw_data_file_path = self.data_ingestion_config.raw_file_path_to_ingest
            train_file_path = self.data_ingestion_config.ingested_train_file_path
            test_file_path = self.data_ingestion_config.ingested_test_data_path
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=get_last_experiment_data(train_file_path),
                                            test_file_path=get_last_experiment_data(test_file_path),
                                            is_ingested=False,
                                            message="Data ingestion was done in last experiments."
                                            )

            last_downloaded_data = get_last_experiment_data(raw_data_file_path)
            self.logger.info(f"Last downloaded data : {last_downloaded_data}")
            is_diff = compare_two_csv(previous_path =last_downloaded_data,current_path= raw_data_file_path, key_columns='default.payment.next.month')
            self.logger.info(f"{'--'*10}is_diff : {is_diff} {'--'*10}")
            if is_diff == "same":
                self.logger.info("Data already ingested")
                return data_ingestion_artifact
            else:
                self.logger.info("Data not ingested yet")
                self.logger.info(f"Reading csv file: [{raw_data_file_path}]")
                raw_data_frame = pd.read_csv(raw_data_file_path)
                raw_data_frame = reduce_mem_usage(raw_data_frame)

                self.logger.info("Splitting data into train and test")
                strat_train_set = None
                strat_test_set = None

                split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

                for train_index, test_index in split.split(raw_data_frame, raw_data_frame["default.payment.next.month"]):
                    strat_train_set = raw_data_frame.loc[train_index]
                    strat_test_set = raw_data_frame.loc[test_index]

                if strat_train_set is not None:
                    self.logger.info(f"Exporting training dataset to file: [{train_file_path}]")
                    os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
                    strat_train_set.to_csv(train_file_path , index=False)
                    train_collection_name = self.data_ingestion_config.ingested_train_collection
                    self.logger.info(f"Exporting training dataset to MongoDB: [{train_collection_name}]")
                    train_conn = MongoDB(train_collection_name, drop_collection=False)
                    status = train_conn.Insert_Many(strat_train_set.to_dict('records'))
                    if status is True:
                        self.logger.info(f"Training dataset exported to MongoDB: [{train_collection_name}]")

                if strat_test_set is not None:
                    self.logger.info(f"Exporting test dataset to file: [{test_file_path}]")
                    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
                    strat_test_set.to_csv(test_file_path , index=False)
                    test_collection_name = self.data_ingestion_config.ingested_test_collection
                    test_conn = MongoDB(test_collection_name, drop_collection=False)
                    status = test_conn.Insert_Many(strat_test_set.to_dict('records'))
                    if status is True:
                        self.logger.info(f"Test dataset exported to MongoDB: [{test_collection_name}]")

                data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                                test_file_path=test_file_path,
                                                                is_ingested=True,
                                                                message="Data ingestion completed successfully."
                                                                )
                self.logger.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
                return data_ingestion_artifact

        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.logger.info(f"{'>>' * 20}Data Ingestion started.{'<<' * 20}")
            data_ingestion_config = self.data_ingestion_config
            dataset_download_url = data_ingestion_config.dataset_download_url
            dataset_download_file_name = data_ingestion_config.dataset_download_file_name
            raw_data_file_path = data_ingestion_config.raw_data_file_path
            self.download_data(dataset_download_url, dataset_download_file_name, raw_data_file_path)

            data_ingestion_response = self.split_data_as_train_test()
            self.logger.info(f"{'>>' * 20}Data Ingestion artifact.{'<<' * 20}")
            self.logger.info(f" Data Ingestion Artifact{data_ingestion_response}")
            self.logger.info(f"{'>>' * 20}Data Ingestion completed.{'<<' * 20}")       
            return data_ingestion_response
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        self.logger.info(f"{'>>' * 20}Data Ingestion log completed.{'<<' * 20} \n\n")


if __name__ == "__main__":
    config = Configuration()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(data_ingestion_config)
    data_ingestion_response = data_ingestion.initiate_data_ingestion()
