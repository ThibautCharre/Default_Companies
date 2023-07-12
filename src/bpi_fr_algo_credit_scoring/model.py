import logging
from typing import Dict
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier as treeC
from sklearn.ensemble import RandomForestClassifier as RfC
from xgboost import XGBClassifier as XgbC

logger = logging.getLogger(__name__)


def models_comparison(
    cleaned_dataset: pd.DataFrame,
    test_size: float = 0.2,
    selected_models: List[str] = ["TREE", "RF", "XGB"],
    random_state: int = 42,
) -> dict:
    """Display predictive scores of a list of Machine Learning algorithms

    Parameters
    ----------
    cleaned_dataset: pd.Dataframe
    test_size: float
    selected_models: List of strings
    random_state: int

    Returns
    -------
    pd.Dataframe

    """
    logger.info(f"Selected algorithms{selected_models}")
    x_train, x_test, y_train, y_test = get_x_y_train_test(
        cleaned_dataset, test_size=test_size, random_state=random_state
    )
    results = get_results(
        x_train, x_test, y_train, y_test, selected_models=selected_models
    )
    logger.info(f"Results{results}")
    return results


def get_x_y_train_test(
    cleaned_dataset: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Split original dataset into training and test datasets

    Parameters
    ----------
    cleaned_dataset: pd.Dataframe
    test_size: float
    random_state: int

    Returns
    -------
    pd.Dataframe (X_train & X_test) and pd.Series (y_train & y_test)

    """
    logger.info("Splitting dataset into training & testing datasets")
    y = cleaned_dataset["X_65"]
    x = cleaned_dataset.drop(columns="X_65")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    logger.info("Split was successful")
    return x_train, x_test, y_train, y_test


def get_results(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train,
    y_test,
    selected_models: List[str] = ["TREE", "RF", "XGB"],
) -> Dict:
    """Calculate predictive scores of ML algorithms

    Parameters
    ----------
    x_train: pd.Dataframe
    x_test: pd.Dataframe
    y_train: pd.Series
    y_test: pd.Series
    selected_models: List of strings

    Returns
    -------
    Dictionary of pd.Dataframe

    """
    logger.info("Calculating accuracy, recall, F1 scores")
    results = {}
    for algo in selected_models:
        if algo == "TREE":
            model_algo = treeC()
        elif algo == "RF":
            model_algo = RfC()
        elif algo == "XGB":
            model_algo = XgbC()

        model_algo.fit(x_train, y_train)
        y_pre = model_algo.predict(x_test)
        acc_sc = accuracy_score(y_test, y_pre)
        recall_sc = recall_score(y_test, y_pre)
        f1_sc = f1_score(y_test, y_pre)
        results[algo] = pd.DataFrame(
            {"Accuracy": [acc_sc], "Recall": [recall_sc], "F1": [f1_sc]}
        )
    logger.info("Calculations done")
    return results
