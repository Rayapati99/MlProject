## Import required packages
import os
import sys


import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object,evaluate_model


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    
    
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation 
        
        '''
        
        try:
            numerical_columns=['reading_score',
                               'writing_score'
                               ]
            
             
            categorical_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch', 
                'test_preparation_course'
                ]
            
            numeric_transformer=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("Scaler",StandardScaler()),
                ]
            )
            
            cat_transformer=Pipeline(
                steps=[
                    
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(handle_unknown='ignore', sparse=False)),
                    ('scaling',StandardScaler())
                    
                    ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            
            col_transformer=ColumnTransformer(transformers=
                [
                    ("numeric_preprocessing",numeric_transformer,numerical_columns),
                    ("cat_preprocessing",cat_transformer,categorical_columns)
                ]
                , remainder='passthrough'
            )
            
            pipeline=Pipeline(steps=
                              [
                                 ("transformer_column",col_transformer) 
                              ])
            
            return pipeline
            
            
            
            
            
        except Exception as e:
            raise CustomException(e,sys)
         
    def initiate_data_transformation(self,train_path,test_path):
        self.train_path=train_path
        self.test_path=test_path         
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            preprocessor_obj=self.get_data_transformer_object()
            
            logging.info("Read train test data completed")
            
            logging.info("obtaining preprocessing object")
            
            target_column_name="math_score"
            
          
            input_feature_train_df=train_df.loc[:, train_df.columns != target_column_name]
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df = test_df.loc[:, test_df.columns != target_column_name]
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"applying preprocessing object on training dataframe and testing dataframe.")
            
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saved Preprocessing object")
            save_object(
                
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
                
            )
            
            return(
                
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
                
                
            )
        
        
        except Exception as e:
            raise CustomException(e,sys)
            

