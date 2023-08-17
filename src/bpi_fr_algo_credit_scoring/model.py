import logging

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


def model_prob_default_prediction(
    training_dataset: pd.DataFrame, y_label: str, new_company_features: pd.DataFrame
) -> float:
    """Calculate probability of default for a given company

    Parameters
    ----------
    training_dataset: pd.DataFrame
    y_label: str
    new_company_features: pd.DataFrame

    Returns
    -------
    float

    """
    logger.info("Importing dataset for calculating default probability prediction")
    model_fitted = model_fitting(cleaned_dataset=training_dataset, y_label=y_label)
    prob_default = model_fitted.predict_proba(new_company_features)
    return prob_default[0][1]


def model_fitting(cleaned_dataset: pd.DataFrame, y_label: str):
    """Fitting of the selected model

    Parameters
    ----------
    cleaned_dataset: pd.Dataframe
    y_label: str

    Returns
    -------
    pd.Dataframe

    """
    logger.info("Fitting of Decision Tree model on imported dataset data")
    y = cleaned_dataset[y_label]
    x = cleaned_dataset.drop(columns=y_label)

    model = DecisionTreeClassifier(random_state=4242)
    model.fit(x, y)

    logger.info("Split was successful")
    return model
