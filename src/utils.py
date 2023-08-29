import pickle
import sys,os
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor



def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train_data,y_train_data,x_test_data,y_test_data):
    try:
        reg = XGBRegressor()
        reg.fit(x_train_data,y_train_data)
        y_pred = reg.predict(x_test_data)
        MAE = mean_absolute_error(y_test_data,y_pred)
        MSE = mean_squared_error(y_test_data,y_pred)
        r2score = r2_score(y_test_data,y_pred)
        return (MAE,MSE,r2score,reg)
    
    except Exception as e:
        logging.info("Error occured at the model training and evaluating step")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Error occured at load_object function in utils")
        raise CustomException(e,sys)
