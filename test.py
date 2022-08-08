from CCdefault.app_pipeline.pipeline import Pipeline
from CCdefault.app_exception.exception import App_Exception
from CCdefault.app_logger import App_Logger
from CCdefault.app_config.configuration import Configuration

import os , sys 

logging = App_Logger("Test_pipeline")
def main():
    try:
        config_path = os.path.join("config","config.yaml")
        pipeline = Pipeline(Configuration(config_file_path=config_path))
        #pipeline.run_pipeline()
        logging.info("Pipeline starting")
        pipeline.start()
        logging.info("main function execution completed.")

    except Exception as e:
        raise App_Exception(e , sys)
        print(e)



if __name__=="__main__":
    main()
