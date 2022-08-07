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


first experiment result :

'train_precision': 0.7549085985104943, 'test_precision': 0.6562922868741543, 'train_recall': 0.42004143906573743, 'test_recall': 0.3654860587792012, 'train_f1': 0.5397555367300012, 'test_f1': 0.46950629235237173, 'train_accuracy': 0.8415416666666666, 'test_accuracy': 0.8173333333333334, 'accuracy_diff': 0.024208333333333276, 'model_accuracy': 0.8294375
