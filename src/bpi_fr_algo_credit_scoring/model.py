from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier as treeC
from sklearn.ensemble import RandomForestClassifier as RfC
from xgboost import XGBClassifier as XgbC


def models_comparison(cleaned_dataset: pd.DataFrame,
                      test_size=0.2,
                      selected_models=['TREE', 'RF', 'XGB'],
                      random_state=42) -> dict:
    """Compare results of Machine Learning Algorithms

    Returns:
        dictionary: _description_
    """
    x_train, x_test, y_train, y_test = get_x_y_train_test(
        cleaned_dataset,
        test_size=test_size,
        random_state=random_state)
    results = get_results(x_train, x_test, y_train, y_test,
                          selected_models=selected_models)
    return results


def get_x_y_train_test(cleaned_dataset: pd.DataFrame,
                       test_size=0.2,
                       random_state=42):
    """Returns X and Y training and tests datasets

    Returns:
        tuples
    """
    y = cleaned_dataset["X_65"]
    x = cleaned_dataset.drop(columns="X_65")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def get_results(x_train, x_test, y_train, y_test,
                selected_models=['TREE', 'RF', 'XGB']) -> Dict:
    """Compare results of Machine Learning Algorithms

    Returns:
        dictionary: _description_
    """
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
        results[algo] = pd.DataFrame({"Accuracy": [acc_sc], "Recall": [recall_sc], "F1": [f1_sc]})
    return results
