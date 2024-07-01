import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import sys,os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj



@dataclass
class DataTransformationconfig:
    preprocessor_ob_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info("In data transformatioin pipeline creation started")

            numerical_columns = ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_columns = ['cut', 'color', 'clarity']

            cut_map = ["Fair","Good","Very Good","Premium","Ideal"]
            clarity_map = ["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"]
            color_map = ["D","E","F","G","H","I","J"]  

            num_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("OrdinalEncoder",OrdinalEncoder(categories=[cut_map,color_map,clarity_map])),
                    ("Scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])

            logging.info("Pipeline completed")
            return preprocessor

            

        except Exception as e:
            logging.info("Error in data transformation pipeline creation process",e)

    def initiate_data_transformation(self,train_data,test_data):
        try:
            train_df = pd.read_csv(train_data)
            test_df = pd.read_csv(test_data)

            logging.info("Reading of train and test data completed")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessor object")

            preprocessing_obj = self.get_data_transformation_object()
            target_column_name = "price"
            drop_columns = [target_column_name,"id"]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applied preprocessor_obj on train and test data")

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessor object as pickle file")
            save_obj (
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
             )

            logging.info("Saving of preprocessor pickle file is completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)
        






