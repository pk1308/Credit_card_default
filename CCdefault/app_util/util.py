import yaml
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_database.mongoDB import MongoDB
from CCdefault.app_logger import App_Logger
from csv_diff import load_csv, compare
import numpy as np
import dill
import pandas as pd
import boto3
import botocore
import os 
import sys
import re 

S3_BUCKET_NAME = "pk1308mlproject"

logging = App_Logger("utils")


def write_yaml_file(file_path: str, data: dict = None):
    """
    Create yaml file 
    file_path: str
    data: dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise App_Exception(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise App_Exception(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array, allow_pickle=True):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise App_Exception(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise App_Exception(e, sys) from e


def save_object(file_path: str, obj):
    """
    file_path: str
    obj: Any sort of object
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise App_Exception(e, sys) from e


def load_object(file_path: str):
    """
    file_path: str
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise App_Exception(e, sys) from e

def load_data_from_mongodb(connection_0bj , limit=4000):
    """
    connection_0bj: mongodb connection object
    """
    try:
        data = connection_0bj.Find_Many( query={}, limit=limit)
        load_df = pd.DataFrame(data)
        if "_id" in load_df.columns:
            load_df.drop(columns=["_id"], inplace=True)
            
        return load_df 
    
    except Exception as e:
        raise App_Exception(e, sys) from e

def s3_download_model(path : str , key_name : str):
    try:
        session = boto3.Session(
        aws_access_key_id= os.environ['AWS_ACCESS_KEY'],
        aws_secret_access_key=os.environ['AWS_ACCESS_SECRET']
        )   
        #Creating S3 Resource From the Session.
        s3 = session.resource('s3')
        bucket = s3.Bucket(S3_BUCKET_NAME)
        obj = bucket.objects.filter()
        file_key = [i for i in obj if key_name in i.key][0]
        bucket.download_file(file_key.key, path) # save to same path
        logging.info("Downloaded Model From S3")

    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise App_Exception(e, sys) from e
        
        
def reduce_mem_usage(df):
    
    try :
        start_mem = df.memory_usage().sum() / 1024**2
        logging.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

        end_mem = df.memory_usage().sum() / 1024**2
        logging.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        logging.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    except Exception as e:
        raise App_Exception(e, sys) from e
    return df

def  get_last_experiment_data(path : str) -> str:
    """_summary_ : This function will return the last experiment file name from the given path

    Args:
        path (str): the path of the current experiment

    Returns:
        str: last experiment file name
    """
    last_file = path
    file_name = os.path.basename(path)
    logging.info(f"get_the_last_experiment_file_name: {path}")
    re_pattern = r"\d{14}"
    dir_name = re.split(re_pattern, path)[0]
    logging.info(f"dir_name: {dir_name}")
    list_of_files = []
    for (root,dirs,files) in os.walk(dir_name, topdown=True):
        if files != []:
            for file in files:
                if file.endswith(file_name):list_of_files.append(os.path.join(root, file))
                
    if path in list_of_files:list_of_files.remove(path)
    if len(list_of_files) != 0:
        
        last_file = max(list_of_files, key=os.path.getmtime)
    return last_file  


def compare_two_csv(current_path : str, previous_path : str  , key_columns : str) -> bool:
    """_summary_ : This function will compare the current and previous csv files and 
    return True if they are same else False 
    Save teh dirrence in previous file dir as yaml"""
    
    diff = compare(current=load_csv(open(current_path), key=key_columns),
        previous= load_csv(open(previous_path), key=key_columns))
    is_diff = sum([len(value) for key, value in diff.items() if value])
    if is_diff or os.path.samefile(current_path,previous_path):
        if is_diff:
            previous_file_name = os.path.basename(previous_path)
            current_file_name = os.path.basename(current_path)
            logging.info(f"{previous_file_name} and {current_file_name} are different")
            diff_path = os.path.join(os.path.dirname(previous_path), f"{previous_file_name}_vs_{current_file_name}_diff.yaml")
            write_yaml_file(data=diff, file_path=diff_path)
            logging.info(f"difference is saved in {diff_path}")
        else:
            logging.info(f"{current_path} and {previous_path} are same file")
        return "Different"
    else:
        return "same"