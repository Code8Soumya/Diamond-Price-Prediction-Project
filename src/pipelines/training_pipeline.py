import os
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    train_arr,test_arr,_ = DataTransformation().initiate_data_transformation(train_data=train_data_path,test_data=test_data_path)
    model_trainer = ModelTrainer().initiate_model_training(train_arr,test_arr)

    
