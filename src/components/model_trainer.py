import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
#from xgboost import XGBRegressor
from src.utils import save_obj,evaluate_model
from dataclasses import dataclass
import sys,os



@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("Splitting dependent and independent variables from train and test array")
            
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Model training started")
            MAE,MSE,r2score,regressor_obj = evaluate_model(x_train,y_train,x_test,y_test)
            logging.info("Model training completed")

            logging.info(f"""The evaluation of the model is as follows:-\n
                             mean_absolute_error: {MAE}\n
                             mean_squared_error: {MSE}\n
                             r2_score: {r2score}
                        """)

            print("The evaluation of the model is as follows:-")
            print("mean_absolute_error: ",MAE)
            print("mean_squared_error: ",MSE)
            print("r2_score:",r2score)

            save_obj (
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=regressor_obj
            )

        except Exception as e:
            logging.info("Error occured at initiate_model_training step")
            raise CustomException(e,sys)