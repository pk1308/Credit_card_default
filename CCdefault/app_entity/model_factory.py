
import importlib
from datetime import datetime
import pandas as pd 
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns 
import yaml
from CCdefault.app_exception.exception import App_Exception
import os
import sys
from collections import namedtuple
from typing import List
from CCdefault.app_logger import App_Logger
from sklearn.metrics import precision_score , recall_score , f1_score , roc_curve , auc  , accuracy_score

logging = App_Logger("model_factory")


sns.set('talk', 'whitegrid', 'dark', font_scale=1.5, font='Ricty',
        rc={"lines.linewidth": 2, 'grid.linestyle': '--'})
GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
@dataclass
class MetricInfoArtifact:
    def __init__(self, experiment_id: str, model_name: str, model_object: object, train_precision: float, test_precision: float,
                 train_recall: float, test_recall: float, train_f1: float, test_f1: float, train_accuracy: float, test_accuracy: float,
                 accuracy_diff: float , model_accuracy : float , model_index : int ):

        self.experiment_id = experiment_id
        self.model_name = model_name
        self.model_object = model_object
        self.train_precision = train_precision
        self.test_precision = test_precision
        self.train_recall = train_recall
        self.test_recall = test_recall
        self.train_f1 = train_f1
        self.test_f1 = test_f1
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.accuracy_diff = accuracy_diff
        self.model_accuracy = model_accuracy
        self.model_index = model_index

def plot_model_report(model_info_artifact: MetricInfoArtifact,
                      false_positive_rate, true_positivity_rate, model_report_dir: str):
    """     Plot the model report 
    """
    logging.info("Plotting the model report")
    data = pd.DataFrame()
    data['train_precision'] = [model_info_artifact.train_precision]
    data['test_precision'] = [model_info_artifact.test_precision]
    data['train_recall'] = [model_info_artifact.train_recall]
    data['test_recall'] = [model_info_artifact.test_recall]
    data['train_f1'] = [model_info_artifact.train_f1]
    data['test_f1'] = [model_info_artifact.test_f1]
    data['train_accuracy'] = [model_info_artifact.train_accuracy]
    data['test_accuracy'] = [model_info_artifact.test_accuracy]
    data['accuracy_diff'] = [model_info_artifact.accuracy_diff]
    model_name = model_info_artifact.model_name
    fpr = false_positive_rate
    tpr = true_positivity_rate
    roc_auc = auc(fpr, tpr)
    if model_report_dir is None:
        model_report_dir = os.path.join(os.getcwd(), 'model_report')
    model_report_file_name = f"{model_name}"
    model_report_file_path = os.path.join(
        model_report_dir, model_report_file_name)
    fig, ax = plt.subplots(1, 2, figsize=(15, 12))
    plt.title(f"Model Report for {model_name}", fontsize=20,
              fontweight='bold', pad=20, loc='center')
    bar = sns.barplot(data=data, ax=ax[0])
    bar.set_xticklabels(bar.get_xticklabels(), rotation=45)
    logging.info(f"ROC AUC Score for {model_name} is {roc_auc}")
    line = sns.lineplot(
        x=fpr, y=tpr, ax=ax[1], color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    line.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    os.makedirs(model_report_dir, exist_ok=True)
    plt.savefig(model_report_file_path)
    # plt.show()
    return model_report_file_path

def evaluate_classification_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                                  X_test: pd.DataFrame, y_test: pd.DataFrame, base_accuracy : float  ,report_dir : str,
                                  estimators: list, is_fitted: bool = False, experiment_id: str = None ) -> List:
    """
      Description:
      This function compare multiple regression model return best model
      Params:
      experiment_id: the experiment id
      estimators: model List 
      X_train: Training dataset input feature
      y_train: Training dataset target feature
      X_test: Testing dataset input feature
      y_test: Testing dataset input feature
      return
      It returned a named tuple

      MetricInfoArtifact ("model_name", "model_object","train_precision", "test_precision",
                                                            "train_recall", "test_recall",
                                                            "train_f1" , "test_f1","model_accuracy", "index_number")
    """
    if experiment_id is None:
        experiment_id = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    model_dir = os.path.join(report_dir, "model_report")
    experiment_dir = os.path.join(model_dir, experiment_id)
    report_dir_list = []
    model_artifact_list = []
    try:
        index = 0 
        best_model = None
        for estimator in estimators:
            model_info_artifact = MetricInfoArtifact(*([None] * 14))
            model_name = estimator.__class__.__name__

            if not  is_fitted:
                logging.info(f"fitting {model_name} model")
                model = estimator.fit(X_train, y_train)
            else:
                logging.info(f"evaluating {model_name} model")
                model = estimator
            model_info_artifact.model_object = model
            logging.info(f"predicting {model_name} model")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_probs = model.predict_proba(X_test)
            y_probs = y_probs[:, 1]
            false_positivity_rate, true_positivity_rate,  _ = roc_curve(
                y_test, y_probs)
            logging.info(f"{model_name} model report")
            model_info_artifact.model_name = model_name
            train_precision = precision_score(y_train, y_train_pred)
            model_info_artifact.train_precision = train_precision
            test_precision = precision_score(y_test, y_test_pred)
            model_info_artifact.test_precision = test_precision
            train_recall = recall_score(y_train, y_train_pred)
            model_info_artifact.train_recall = train_recall
            test_recall = recall_score(y_test, y_test_pred)
            model_info_artifact.test_recall = test_recall
            train_f1 = f1_score(y_train, y_train_pred)
            model_info_artifact.train_f1 = train_f1
            test_f1 = f1_score(y_test, y_test_pred)
            model_info_artifact.test_f1 = test_f1
            test_accuracy = accuracy_score(y_true=y_test, y_pred=y_test_pred)
            train_accuracy = accuracy_score(y_true= y_train , y_pred = y_train_pred)
            model_info_artifact.test_accuracy = test_accuracy
            model_info_artifact.train_accuracy = train_accuracy
            model_accuracy  = (train_accuracy + test_accuracy) / 2
            model_info_artifact.model_accuracy = model_accuracy
            diff_test_train_acc= abs(train_accuracy - test_accuracy)
            model_info_artifact.accuracy_diff = diff_test_train_acc
            model_info_artifact.model_index = index
            model_artifact_list.append(model_info_artifact)
            os.makedirs(experiment_dir, exist_ok=True)
            logging.info(f"Plotting {model_name} model report")
            logging.info(f"{model_info_artifact.__dict__}")
            report_path = plot_model_report(
                model_info_artifact, false_positivity_rate, true_positivity_rate, experiment_dir )
            logging.info(f"Model report saved at {report_path}")
            report_dir_list.append(report_path)
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.10:
                base_accuracy = model_accuracy
                best_model = model_info_artifact
                logging.info(f"Acceptable model found {model_info_artifact}. ")
            index += 1
        if best_model is None:
            logging.info("No acceptable model found")
    except Exception as e:
        raise App_Exception(e, sys)
    return best_model

            


# def evaluate_regression_model(model_list: list, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
#                               y_test: np.ndarray, base_accuracy: float = 0.6) -> MetricInfoArtifact:
#     """
#     Description:
#     This function compare multiple regression model return best model

#     Params:
#     model_list: List of model
#     X_train: Training dataset input feature
#     y_train: Training dataset target feature
#     X_test: Testing dataset input feature
#     y_test: Testing dataset input feature

#     return
#     It retured a named tuple
    
#     MetricInfoArtifact = namedtuple("MetricInfo",
#                                 ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
#                                  "test_accuracy", "model_accuracy", "index_number"])

#     """
#     try:

#         index_number = 0
#         metric_info_artifact = None
#         for model in model_list:
#             model_name = str(model)  # getting model name based on model object
#             logging.info(f"{'>>' * 30}Started evaluating model: [{type(model).__name__}] {'<<' * 30}")

#             # Getting prediction for training and testing dataset
#             y_train_pred = model.predict(X_train)
#             y_test_pred = model.predict(X_test)

#             # Calculating r squared score on training and testing dataset
#             train_acc = r2_score(y_train, y_train_pred)
#             test_acc = r2_score(y_test, y_test_pred)

#             # Calculating mean squared error on training and testing dataset
#             train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#             test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

#             # Calculating harmonic mean of train_accuracy and test_accuracy
#             model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
#             diff_test_train_acc = abs(test_acc - train_acc)

#             # logging all important metric
#             logging.info(f"{'>>' * 30} Score {'<<' * 30}")
#             logging.info(f"Train Score\t\t Test Score\t\t Average Score")
#             logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

#             logging.info(f"{'>>' * 30} Loss {'<<' * 30}")
#             logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].")
#             logging.info(f"Train root mean squared error: [{train_rmse}].")
#             logging.info(f"Test root mean squared error: [{test_rmse}].")

#             # if model accuracy is greater than base accuracy and train and test score is within certain thershold
#             # we will accept that model as accepted model
#             if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
#                 base_accuracy = model_accuracy
#                 metric_info_artifact = MetricInfoArtifact(model_name=model_name,
#                                                           model_object=model,
#                                                           train_rmse=train_rmse,
#                                                           test_rmse=test_rmse,
#                                                           train_accuracy=train_acc,
#                                                           test_accuracy=test_acc,
#                                                           model_accuracy=model_accuracy,
#                                                           index_number=index_number)

#                 logging.info(f"Acceptable model found {metric_info_artifact}. ")
#             index_number += 1
#         if metric_info_artifact is None:
#             logging.info(f"No model found with higher accuracy than base accuracy")
#         return metric_info_artifact
#     except Exception as e:
#         raise App_Exception(e, sys) from e


def get_sample_model_config_yaml_file(export_dir: str):
    try:
        model_config = {
            GRID_SEARCH_KEY: {
                MODULE_KEY: "sklearn.model_selection",
                CLASS_KEY: "GridSearchCV",
                PARAM_KEY: {
                    "cv": 3,
                    "verbose": 1
                }

            },
            MODEL_SELECTION_KEY: {
                "module_0": {
                    MODULE_KEY: "module_of_model",
                    CLASS_KEY: "ModelClassName",
                    PARAM_KEY:
                        {"param_name1": "value1",
                         "param_name2": "value2",
                         },
                    SEARCH_PARAM_GRID_KEY: {
                        "param_name": ['param_value_1', 'param_value_2']
                    }

                },
            }
        }
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, 'w') as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise App_Exception(e, sys)


class ModelFactory:
    def __init__(self, model_config_path: str = None, ):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None

        except Exception as e:
            raise App_Exception(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise App_Exception(e, sys) from e

    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise App_Exception(e, sys) from e

    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command: from {module} import {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise App_Exception(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        execute_grid_search_operation(): function will perform parameter search operation, and
        it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: you're all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class

            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            message = f'{">>" * 30} f"Training {type(initialized_model.model).__name__} Started." {"<<" * 30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>" * 30} f"Training {type(initialized_model.model).__name__}" completed {"<<" * 30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            message = f'{">>" * 30} f"Training {type(initialized_model.model).__name__} Finished" {"<<" * 30}'
            logging.info(message)
            logging.info(f'{">>" * 10} Best model: {">>" * 10}')
            logging.info(grid_searched_best_model.best_model)
            logging.info(f'{">>" * 10} Best Score: {">>" * 10}')
            logging.info(grid_searched_best_model.best_score)
            logging.info(f'{">>" * 10} Best Params: {">>" * 10}')
            logging.info(grid_searched_best_model.best_parameters)

            return grid_searched_best_model
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function will return a list of model details.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY]
                                                            )
                model = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                  property_data=model_obj_property_data)

                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform paramter search operation and
        it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise App_Exception(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise App_Exception(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise App_Exception(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_accuracy=0.6
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_accuracy < grid_searched_best_model.best_score:
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_accuracy = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise App_Exception(e, sys) from e

    def get_best_model(self, X, y, base_accuracy=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        except Exception as e:
            raise App_Exception(e, sys)
