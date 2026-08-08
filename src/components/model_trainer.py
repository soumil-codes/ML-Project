import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbour": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "positive": [True, False]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "max_depth": [None, 3, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },

                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "K-Nearest Neighbour": {
                    "n_neighbors": [3, 5, 7, 9, 11, 15],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },

                "XGBRegressor": {
                    "n_estimators": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },

                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },

                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                    "loss": ["linear", "square", "exponential"]
                }
            }
            
            tuned_models = {}
            tuned_model_scores = {}

            for model_name, model in models.items():

                logging.info(f"Hyperparameter tuning started for {model_name}")

                # Check if parameters are defined for this model
                if model_name not in params:
                    logging.info(f"No parameters found for {model_name}. Skipping.")
                    continue

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[model_name],
                    n_iter=20,
                    scoring='r2',
                    cv=5,
                    verbose=1,
                    random_state=42,
                    n_jobs=-1
                )

                random_search.fit(X_train, y_train)

                # Best model found by RandomizedSearchCV
                best_model = random_search.best_estimator_

                tuned_models[model_name] = best_model

                logging.info(
                    f"Best parameters for {model_name}: "
                    f"{random_search.best_params_}"
                )

                logging.info(
                    f"Best CV R2 score for {model_name}: "
                    f"{random_search.best_score_}"
                )

                # Evaluate tuned model on test set
                predicted = best_model.predict(X_test)

                test_score = r2_score(y_test, predicted)

                tuned_model_scores[model_name] = test_score

                logging.info(
                    f"Test R2 score for tuned {model_name}: "
                    f"{test_score}"
                )
                
            best_model_name = max(
                tuned_model_scores,
                key=tuned_model_scores.get
            )

            best_model_score = tuned_model_scores[best_model_name]

            best_model = tuned_models[best_model_name]

            logging.info(
                f"Final Best Model: {best_model_name}"
            )

            logging.info(
                f"Final Test R2 Score: {best_model_score}"
            )

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)