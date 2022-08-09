Creating conda environment

```

conda create -p venv python==3.7 -y

```

```

conda activate venv/

```

```
pip install -r requirements.txt

```

```
dvc init

```

CONNECTION_STRING = os.getenv('MONGODB_CONNSTRING')
MONGODB_CONNSTRING="mongodb+srv://USERNAME:PASSWORD@cluster0.wzb80.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"

**base Model**

<img src="CCdefault/app_artifact/stage03_model_training/model_report/20220809140111/cluster_custom_model/model_report/2022-08-09-14-17-19/EstimatorModel.png" alt="Alt text" title="Pca with Smote Best Model">

**Pca with Smote Best Model**

<img src="CCdefault/app_artifact/stage03_model_training/model_report/20220809114247/Base_model/model_report/2022-08-09-11-55-48/SVC.png" alt="Alt text" title="Pca with Smote Best Model">

**Smote Model is over fitted**

Clustered Model

<img src="CCdefault/app_artifact/stage03_model_training/model_report/20220809121752/cluster_custom_model/model_report/2022-08-09-12-45-41/EstimatorModel.png" alt="Alt text" title="Smote Best Model">

Base best model

<img src="CCdefault/app_artifact/stage03_model_training/model_report/20220809121752/Base_model/model_report/2022-08-09-12-30-31/CatBoostClassifier.png" alt="Alt text" title="Smote Best Model">

**PCA model with 14 comp**
clustered model

<img src="CCdefault/app_artifact/stage03_model_training/model_report/20220809131614/cluster_custom_model/model_report/2022-08-09-13-33-13/EstimatorModel.png" alt="Alt text" title="Pca Best Model">
