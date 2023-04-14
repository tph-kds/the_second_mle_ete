import os 
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from   catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor , GradientBoostingRegressor , RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_objective , evaluate_model

@dataclass
class DataTrainerConfig:
    trained_model_file_path = os.path.join("artifacts" , "model.pkl")

class DataTrainer:
    def __init__(self) :
        self.model_trainer_config = DataTrainerConfig()



    def initiate_model_trainer(self , train_array , test_array ):
        try:
            logging.info("Split training and test input data")
            X_train , y_train , X_test , y_test = (
                train_array[: , :-1] , 
                train_array[: , -1] , 
                test_array[: , :-1]  ,  
                test_array[: , -1]  , 

            )

            models = {
                "Random Forest" : RandomForestRegressor() , 
                "Decision Tree" : DecisionTreeRegressor() , 
                "Gradient Boosting" : GradientBoostingRegressor() , 
                "Linear Regression" : LinearRegression() , 
                "K-Neighbors Classifier" : KNeighborsRegressor() , 
                "XGBClassifier" : XGBRegressor() , 
                "CatBoosting Classifier" : CatBoostRegressor(verbose= False) ,
                "AdaBoost Classifier" : AdaBoostRegressor() ,  
            }

            model_report:dict = evaluate_model(X_train = X_train  , y_train = y_train , X_test = X_test , y_test = y_test ,  models = models)
            

            ## To get best model score from dict 
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            if best_model_score < 0.6 :
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing data")

            save_objective(file_path= self.model_trainer_config.trained_model_file_path , 
            obj= best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test  ,  predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e , sys)