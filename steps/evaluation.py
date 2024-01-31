import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2Score, RMSE
from typing import Tuple
from typing_extensions import Annotated

@step
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame,
                   ) -> Tuple[
                       Annotated[float, "r2_score"],
                       Annotated[float, "rmse"],
                       Annotated[float, "mse"]
                   ]:
    """
    Evaluates the model on the ingested data.
    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)

        r2_class = R2Score()
        r2_score = r2_class.calculate_scores(y_test, prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)

        return r2_score, rmse, mse
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        raise e