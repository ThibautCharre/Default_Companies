## Algo for credit scoring

## Road map
1/ Importing dataset
2/ Cleaning dataset and missing values
3/ Feature engineering phase
4/ Machine Learning algorithms predictions
5/ Selecting the best algo and optimising its hyperparameters

## Description
The project aims at detecting companies that are most likely to default.
Machine learning algorithms are imported and predictive scores are calculated and compared
to select the most efficient and reliable model.
The original dataset consists of financial variables issued from companies' financial reports
and a labeled variable (X_65) indicating historical default (1) or the non-default events (0) of companies.

The algorithm delivered imports the original dataset, cleans it and displays a ML algorithms scores such as
precision, recall and F1-score.
Hypertuning of selected algorithm parameters is made in a Jupyter notebook located in the folder "/notebooks"
while variables shap values are described in another Jupyter notebook still located in the same folder.

## Todo :
- Poetry: done
- Pre-commit: done
- Pytest: done