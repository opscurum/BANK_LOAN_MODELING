# BANK_LOAN_MODELING
(Powerful data combined with 3 powerful machine learning methods.)


This repository contains code for modeling bank loans using Python. The code is designed to preprocess the data, clean it, and apply different modeling techniques to predict loan statuses. 

## Getting Started

To get started, follow these steps:

1. Install the required dependencies by running `pip install pandas numpy seaborn missingno scikit-learn xgboost matplotlib` in your terminal.
2. Clone the repository: `git clone https://github.com/your-username/bank-loan-modeling.git`.
3. Navigate to the project directory: `cd bank-loan-modeling`.

## Prerequisites

To run the code, you need to have Python installed on your machine. The code is compatible with Python 3.x.

## Usage

1. Open the Python file `bank_loan_modeling.py` in your preferred code editor.
2. Modify the file path in the line `df = pd.read_csv("loans_2007.csv")` to point to your dataset file.
3. Run the script to execute the code.

## Description

The code performs the following steps:

1. Imports necessary libraries such as pandas, numpy, seaborn, etc.
2. Loads the dataset using pandas.
3. Performs data preprocessing and cleaning, including removing unnecessary columns, handling missing values, converting data types, etc.
4. Visualizes the missing data structure using the `missingno` library.
5. Handles multivariate outlier observations using the Local Outlier Factor (LOF) method.
6. Converts object data to numeric data by applying encoding techniques and creating dummy variables.
7. Builds a logistic regression model and evaluates its performance using various metrics such as confusion matrix, accuracy score, classification report, and ROC curve.
8. Splits the dataset into training and testing sets and performs cross-validation to estimate the model's performance.
9. Outputs the accuracy score and cross-validation scores.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contributing

Contributions to this project are welcome. Please open an issue or submit a pull request with your proposed changes.

## Acknowledgments

The code in this repository is based on the tutorials and examples from various sources. Thanks to the authors of those resources for their contributions to the field.

