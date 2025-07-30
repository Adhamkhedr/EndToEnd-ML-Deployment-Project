from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.utils.exception_handling import CustomException
from src.utils.logging import get_logger
import sys

logger = get_logger(__name__)

#Take y_true (actual target values) and y_pred (predicted values)
#Return 3 scores ➜ MAE, RMSE, R²
def evaluate_model(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
#Mean Squared Error (MSE) if squared=True (default) AND the Root Mean Squared Error (RMSE) → if squared=False


        logger.info(f"Evaluation metrics - R2: {r2}, MAE: {mae}, RMSE: {rmse}")
        return {
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        }
    except Exception as e:
        logger.error("Exception occurred during model evaluation.")
        raise CustomException(e, sys)
