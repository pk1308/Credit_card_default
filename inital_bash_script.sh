mkdir CCdefault
cd CCdefault

echo "Createing project structure."
mkdir app_config app_exception app_database app_logger app_src app_util config app_entity app_artifcats app_notebook 
echo "Project structure created."

echo "started creating python script for each module."
touch app_entity/__init__.py app_entity/config_entity.py app_entity/artifacts_entity.py 
echo "app configuration file created successfully."


echo "started creating python script for each module."
touch app_config/__init__.py app_config/configuration.py 
echo "app configuration file created successfully."

echo "Started creating app exception scripts."
touch app_exception/__init__.py app_exception/exception.py 
echo "App exception script created."

echo "Started creating app database scripts."
touch app_database/__init__.py app_database/mongoDB.py 
echo "App exception script created."

echo "Started creating app logger scripts."
touch app_logger/__init__.py app_logger/logger.py
echo "App logger script created."

echo "Started creating app pipeline scripts."
touch app_pipeline/__init__.py app_pipeline/training_pipeline.py  app_pipeline/prediction_pipeline.py
echo "App pipeline script created."

echo "Started creating app_src scripts."
touch app_src/__init__.py app_src/stage_00_data_ingestion.py app_src/stage_01_data_validation.py 
touch app_src/stage_02_data_transformation.py app_src/stage_03_model_trainer.py app_src/stage_04_model_evalution.py
echo "App src scrits created successfully."


echo "Started creating app_util scripts."
touch app_util/__init__.py app_util/util.py
echo "App util file created successfully."

cd ..

echo "Configuration file creating.."

mkdir config
touch config/config.yaml touch config/model.yaml touch config/schema.yaml


echo "Configuration file created successfully."