# src/models/xgb_model.py
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    make_scorer
)
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def train_xgb_multioutput(
    X_train,
    y_train,
    param_grid,
    random_state=42,
    cv=3,
    verbose=3
):
    """
    Train MultiOutput XGBRegressor with GridSearchCV
    """

    base_xgb = XGBRegressor(
        learning_rate=0.01,
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist"
    )

    model = MultiOutputRegressor(base_xgb)

    scoring = {
        "R2": "r2",
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "MSE": make_scorer(mean_squared_error, greater_is_better=False),
        "RMSE": make_scorer(
            lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
            greater_is_better=False
        ),
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        refit="R2",
        cv=cv,
        verbose=verbose,
        n_jobs=1
    )

    grid.fit(X_train, y_train)

    return grid
