import logging

import arff
import pandas as pd

logger = logging.getLogger(__name__)


def read_yearly_data(path: str, default_year: int = 1) -> pd.DataFrame:
    """

    Parameters
    ----------
    path: String character indicating directory of data file
    default_year: Integer representing default year of companies (default is 1)

    Returns
    -------
    Dataframe containing data of default and non-default companies

    """
    logger.info(f"Importing dataset for {default_year} year default companies")
    dt = pd.DataFrame(
        data=arff.load(
            open(file=f"{path}/{default_year}year.arff", mode="r", encoding="utf-8")
        )["data"],
        columns=[f"X_{n_col}" for n_col in range(1, 66)],
    )
    dt["X_65"] = dt["X_65"].astype("int")
    logger.info(f"Dataset correctly imported")
    dt = rename_columns(dt)
    logger.info("Columns renamed")
    return dt


def rename_columns(default_risk_dataset: pd.DataFrame) -> pd.DataFrame:
    """Renaming of dataset columns

    Parameters
    ----------
    default_risk_dataset: pd.DataFrame

    Returns
    -------
    pd.DataFrame

    """
    variable_dict = {
        "X_1": "net profit / total assets",
        "X_2": "total liabilities / total assets",
        "X_3": "working capital / total assets",
        "X_4": "current assets / short - term liabilities",
        "X_5": "[(cash + short - term securities + receivables - short-term liabilities) / "
        "(operating expenses - depreciation)] * 365",
        "X_6": "retained earnings / total assets",
        "X_7": "EBIT / total assets",
        "X_8": "book value of equity / total liabilities",
        "X_9": "sales / total assets",
        "X_10": "equity / total assets",
        "X_11": "(gross profit + extraordinary items + financial expenses) / total assets",
        "X_12": "gross profit / short - term liabilities",
        "X_13": "(gross profit + depreciation) / sales",
        "X_14": "(gross profit + interest) / total assets",
        "X_15": "(total liabilities * 365) / (gross profit + depreciation)",
        "X_16": "(gross profit + depreciation) / total liabilities",
        "X_17": "total assets / total liabilities",
        "X_18": "gross profit / total assets",
        "X_19": "gross profit / sales",
        "X_20": "(inventory * 365) / sales",
        "X_21": "sales(n) / sales(n - 1)",
        "X_22": "profit on operating activities / total assets",
        "X_23": "net profit / sales",
        "X_24": "gross profit( in 3 years) / total assets",
        "X_25": "(equity - share capital) / total assets",
        "X_26": "(net profit + depreciation) / total liabilities",
        "X_27": "profit on operating activities / financial expenses",
        "X_28": "working capital / fixed assets",
        "X_29": "logarithm of total assets",
        "X_30": "(total liabilities - cash) / sales",
        "X_31": "(gross profit + interest) / sales",
        "X_32": "(current liabilities * 365) / cost of products sold",
        "X_33": "operating expenses / short - term liabilities",
        "X_34": "operating expenses / total liabilities",
        "X_35": "profit on sales / total assets",
        "X_36": "total sales / total assets",
        "X_37": "(current assets - inventories) / long - term liabilities",
        "X_38": "constant capital / total assets",
        "X_39": "profit on sales / sales",
        "X_40": "(current assets - inventory - receivables) / short - term liabilities",
        "X_41": "total liabilities / ((profit on operating activities + depreciation) * (12 / 365))",
        "X_42": "profit on operating activities / sales",
        "X_43": "rotation receivables + inventory turnover in days",
        "X_44": "(receivables * 365) / sales",
        "X_45": "net profit / inventory",
        "X_46": "(current assets - inventory) / short - term liabilities",
        "X_47": "(inventory * 365) / cost of products sold",
        "X_48": "EBITDA(profit on operating activities - depreciation) / total assets",
        "X_49": "EBITDA(profit on operating activities - depreciation) / sales",
        "X_50": "current assets / total liabilities",
        "X_51": "short - term liabilities / total assets",
        "X_52": "(short - term liabilities * 365) / cost of products sold)",
        "X_53": "equity / fixed assets",
        "X_54": "constant capital / fixed assets",
        "X_55": "working capital",
        "X_56": "(sales - cost of products sold) / sales",
        "X_57": "(current assets - inventory - short - term liabilities) / (sales - gross profit - depreciation)",
        "X_58": "total costs / total sales",
        "X_59": "long - term liabilities / equity",
        "X_60": "sales / inventory",
        "X_61": "sales / receivables",
        "X_62": "(short - term liabilities * 365) / sales",
        "X_63": "sales / short - term liabilities",
        "X_64": "sales / fixed assets",
        "X_65": "Default",
    }
    default_risk_dataset.rename(mapper=variable_dict, axis=1, inplace=True)
    return default_risk_dataset
