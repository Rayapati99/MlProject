import os
import sys
import dill
import pandas as pd

from src.utils import save_object

from src.utils import evaluate_model,load_object

from src.exception import CustomException
from src.logger import logging

class predictpipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            
            print("before loading")
            
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("after loading")
            
            data_scaled=preprocessor.transform(features)
            pred=model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
    ## math_score, reading_score, writing_score
    ### gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course
class CustomData:
    def __init__(self,reading_score:int, writing_score:int,
     gender:str, race_ethnicity:str, parental_level_of_education:str, lunch:str, test_preparation_course:str):
        self.reading_score=reading_score
        self.writing_score=writing_score
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        
        
    def get_data_as_data_frame(self):
        try:
            
            custom_data_input_dict={
                
                "lunch":[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'parental_level_of_education':[self.parental_level_of_education],
                'race_ethnicity':[self.race_ethnicity],
                'gender':[self.gender],
                'writing_score':[self.writing_score],
                'reading_score':[self.reading_score]
                }
                
            return pd.DataFrame(custom_data_input_dict)
                
        except Exception as e:
            raise CustomException(e,sys)
    