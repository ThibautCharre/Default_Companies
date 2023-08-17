## Algo for credit scoring

## Road map
1/ Importing dataset
2/ Cleaning of dataset and treatment of missing values
3/ Fitting of the ML seleceted model to the training dataset
4/ Import of new company values to calculate a probability of default
5/ Probability calculation

## Description
The project aims at detecting companies that are most likely to default.
Original dataset contains 65 financial variables describing companies financial statements.
Research regarding importances of variables are realized in the Jupyter notebooks folder.
Besides, comparison of ML models aiming at selecting the best fitted model is done within the same folder.
The principal algorithm is located in the src/<package>/main.py file under the function run()

## Main Algo launch
To run the algo aiming at calculating a default probability for a new company, follow these instructions:

1/ Retrieve the project through git
git pull origin

2/ After having installed poetry, install the environment and the packages needed
by executing the following command in a powershell
poetry install

3/ To change values of a company launched in the main algo, modify values from the csv file "new_comp_data.csv",
located in "bpi-fr-algo-credit-scoring\tests\data"

4/ By running the command
poetry run app_script
