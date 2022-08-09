import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from CCdefault.app_entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact, \
    DataTransformationArtifact
from CCdefault.app_entity.config_entity import DataTransformationConfig
from CCdefault.app_logger import App_Logger
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_util.util import read_yaml_file, save_object, reduce_mem_usage , save_numpy_array_data
from CCdefault.app_constants import *


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """custom feature generator class to generate cluster class for the data
    scaler : StandardScaler clustering using kmeans++ and kneed"""

    def __init__(self, pay_x_columns , Age_column ,bil_amt_columns , pay_amt_columns,limit_bin, encoder= OneHotEncoder(sparse=False)):
        try:
            self.cluster = None
            self.logger = App_Logger("Feature_generator")
            self.pay_x = pay_x_columns
            self.age = Age_column
            self.bill_amt = bil_amt_columns
            self.pay_amt_columns = pay_amt_columns
            self.limit_bin = limit_bin
            self.encoder = encoder

        except Exception as e:
            raise App_Exception(e, sys) from e

    def fit(self, X, y=None):
        data = X.copy()
        data = pd.DataFrame()
        pay_feature = lambda x: x if x < 4 else 4
        for col in self.pay_x:
            data[col] = X[col].apply(pay_feature)
        data[self.age]= pd.cut(X[self.age],[20, 25, 30, 35, 40, 50, 60, 80])
        for col in self.bill_amt:
            data[col] = pd.cut(X[col],[-350000,-1,0,25000, 75000, 200000, 2000000])
        for col in self.pay_amt_columns:
            data[col] = pd.cut(X[col],[-1, 0, 25000, 50000, 100000, 2000000])
        data[self.limit_bin] =pd.cut(X[self.limit_bin],[5000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1100000])
        [data[col].astype("category")for col in data.columns]
        data_encoded = self.encoder.fit_transform(data)
        wcss=[]
        for i in range(1,11):
            kmeans=KMeans(n_clusters=i, init='k-means++',random_state=42)
            kmeans.fit(data_encoded)
            wcss.append(kmeans.inertia_) 

        kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
        total_clusters=kn.knee
        self.logger.info(f"total cluster :{total_clusters}")
        self.cluster = KMeans(n_clusters=total_clusters, init='k-means++',random_state=42)
        self.cluster.fit(data_encoded)
        return self
    
    def transform(self, X, y=None):
        try:
            self.logger.info("Transforming data")
            data = pd.DataFrame()
            pay_feature = lambda x: x if x < 4 else 4
            for col in self.pay_x:
                data[col] = X[col].apply(pay_feature)
            data[self.age]= pd.cut(X[self.age],[20, 25, 30, 35, 40, 50, 60, 80])
            for col in self.bill_amt:
                data[col] = pd.cut(X[col],[-350000,-1,0,25000, 75000, 200000, 2000000])
            for col in self.pay_amt_columns:
                data[col] = pd.cut(X[col],[-1, 0, 25000, 50000, 100000, 2000000])
            data[self.limit_bin] =pd.cut(X[self.limit_bin],[5000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1100000])
            [data[col].astype("category")for col in data.columns]
            data_encoded = self.encoder.transform(data)
            cluster  = self.cluster.predict(data_encoded)
            generated_feature = np.c_[data_encoded , cluster]
            return generated_feature
        except Exception as e:
            raise App_Exception(e, sys) from e

class DataTransformation:
    """Data transformation class . Choose the columns to model and transform the data"""

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        try:
            self.logger = App_Logger("Data Transformation")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.logger.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            
            pay_x_columns = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            bill_amt_columns = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            pay_amt_columns = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            Age_columns = "AGE"
            limit_columns = 'LIMIT_BAL'
            preprocessing = Pipeline(steps=[('feature_generator', FeatureGenerator(pay_amt_columns=pay_amt_columns, bil_amt_columns=bill_amt_columns,
                                                                       pay_x_columns=pay_x_columns,Age_column= Age_columns,
                                                                      limit_bin=limit_columns))])
            return preprocessing

        except Exception as e:
            raise App_Exception(e, sys) from e
    def over_sample_input(self , df , target_columns ):
        try :
            features = df.drop(target_columns , axis=1)
            target = df[target_columns]
            scaler = StandardScaler()
            scaler.fit(features)
            features_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns)
            smote_model = SMOTE(sampling_strategy='minority', random_state=1965, k_neighbors=3)
            over_sampled_trainX, over_sampled_trainY = smote_model.fit_resample(X=features_scaled, y=target)
            features_over_sampled = pd.DataFrame(scaler.inverse_transform(over_sampled_trainX,) , columns=features.columns)
            return features_over_sampled , over_sampled_trainY
        except Exception as e:
            raise App_Exception(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            self.logger.info("Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_object()
            schema_file_path = self.data_validation_artifact.schema_file_path

            dataset_schema = read_yaml_file(file_path=schema_file_path)
            target_column_name = dataset_schema[TARGET_COLUMN_KEY]
            self.logger.info(f"Target column name : {target_column_name}")
            columns_to_cluster = dataset_schema[COLUMNS_TO_CLUSTER_KEY]
            self.logger.info("Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            self.logger.info("Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path , usecols=columns_to_cluster)
            train_df = reduce_mem_usage(train_df)
            test_df = pd.read_csv(test_file_path , usecols=columns_to_cluster)
            test_df = reduce_mem_usage(test_df)
        

            self.logger.info("Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            
            self.logger.info("Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)
        
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")
            
            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            self.logger.info("Saving transformed training and testing array.")
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path
            os.makedirs(transformed_test_dir, exist_ok=True)
            os.makedirs(transformed_train_dir, exist_ok=True)
            os.makedirs(os.path.dirname(preprocessing_obj_file_path), exist_ok=True)
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)
            self.logger.info("Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path, obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfully.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path

                                                                      )
            self.logger.info(f"Data transformations artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise App_Exception(e, sys) from e

    def __del__(self):
        self.logger.info(f"{'>>' * 30}Data Transformation log completed.{'<<' * 30} \n\n")