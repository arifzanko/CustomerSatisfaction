import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluation(ABC):
    """
    Abstract class defining strategy for evaluation our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model

        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            None
        """
        pass

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        

class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score: float
        """
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            r2 = r2_score(y_true, y_pred)
            logging.info("The r2 score value is: " + str(r2))
            return r2
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message:  "
                + str(e)
            )
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE)
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse: float
        """
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message:  "
                + str(e)
            )
            raise e