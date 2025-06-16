import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR 
from sklearn.tree import DecisionTreeRegressor 
from xgboost import XGBRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() 

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "XGBRegressor": XGBRegressor()
            }

            model_report:dict = evaluate_models(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models
            )

            # Get the best model based on the highest score
            best_model_score = max(sorted(model_report.values()))

            # Get the best model from the models dictionary
            #best_model_name = list(model_report.keys())[list(model_report.values().index(best_model_score))]
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient score", sys)
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predicted)
            return r2_squared 
        
        except Exception as e:
            raise CustomException(e, sys)
        
    

    