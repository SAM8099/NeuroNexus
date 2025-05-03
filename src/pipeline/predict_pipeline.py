import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self, 
                 pclass: int,
                 sex: str,
                 age: float,
                 sibsp: int,
                 parch: int,
                 fare: float,
                 embarked: str):
        self.pclass = pclass
        self.sex = sex
        self.age = age
        self.sibsp = sibsp
        self.parch = parch
        self.fare = fare
        self.embarked = embarked
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Pclass': [self.pclass],
                'Sex': [self.sex],
                'Age': [self.age],
                'SibSp': [self.sibsp],
                'Parch': [self.parch],
                'Fare': [self.fare],
                'Embarked': [self.embarked]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame created from custom input data")
            return df
            
        except Exception as e:
            logging.error(f"Exception occurred in CustomData: {e}")
            raise CustomException(e, sys)


