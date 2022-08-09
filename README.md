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

**Pca with Smote Best Model**

```
![Pca with Smote Best Model](CCdefault/app_artifact/stage03_model_training/model_report/20220809114247/cluster_custom_model/model_report/2022-08-09-12-10-13/BaseModel.png  "Best Model")
```

**Smote Model is over fitted**

```
![Smote Clustered model](CCdefault/app_artifact/stage03_model_training/model_report/20220809121752/cluster_custom_model/model_report/2022-08-09-12-45-41/EstimatorModel.png  "Smote Clustered model")
```

```
![Smote Base model](CCdefault/app_artifact/stage03_model_training/model_report/20220809121752/cluster_custom_model/model_report/2022-08-09-12-45-41/BaseModel.png "Smote Base model")
```