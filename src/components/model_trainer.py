from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pickle
import numpy as np
import os
from dataclasses import dataclass
from src.utils.exception_handling import CustomException
from src.utils.logging import get_logger
import sys
from src.components.model_evaluation import evaluate_model  #I will build in next file

logger = get_logger(__name__)   # Initialize logger

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")  #Create a config object with one attribute:
    #the path where the model will be saved.

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()   #gives class access to the config path.

    """Accepts train_arr and test_arr
    Splits them into X (features) and y (target)
    Trains a model
    Evaluates it
    Saves it """

    def initiate_model_training(self, train_array, test_array):
        try:
            logger.info("Splitting training and testing arrays into features and targets")
            X_train = train_array[:, :-1]    
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]
            #Take all columns except the last one as input (X)
            #Take only the last column as the target (y)

            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor()
            }

            #Track the performance of each model ➜ so you remember how each one did
            #Keep track of the current “best” model ➜ so that in the end you know which to save

            model_report = {}         
            # empty dictionary to store the evaluation results (e.g., MAE, RMSE, R²) for each model.

            best_score = float("-inf")  # store best R² score so far
            best_model = None           # store the best model object
            best_model_name = None      # store the name of the best model

            for name, model in models.items():    #Loop over each model
                logger.info(f"Training and evaluating model: {name}")
                model.fit(X_train, y_train)          # train the model
                y_pred = model.predict(X_test)      # make predictions
                scores = evaluate_model(y_test, y_pred)  # get MAE, RMSE, R² from the evaluate_model()
                #that will be created in the model_evaluate file

                print(f"\n{name} scores:")
                print(f"  R²   : {scores['R2']:.4f}")
                print(f"  MAE  : {scores['MAE']:.4f}")
                print(f"  RMSE : {scores['RMSE']:.4f}")

                model_report[name] = scores
                if scores["R2"] > best_score:    
                    best_score = scores["R2"]
                    best_model = model     
                    best_model_name = name

            # Save best model
            logger.info(f"Saving best model: {best_model_name} with R² score: {best_score}")
            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            print("\n=== Final Model Comparison Report ===")
            for name, score in model_report.items():
                print(f"{name}: R²={score['R2']:.4f}, MAE={score['MAE']:.4f}, RMSE={score['RMSE']:.4f}")
            
            logger.info("Model training completed successfully")

            return {
                "best_model": best_model_name,
                "best_score_r2": best_score,
                "model_path": self.config.trained_model_file_path,
                "all_scores": model_report
            }

        except Exception as e:
            logger.error("Exception occurred during model training")
            raise CustomException(e, sys)
