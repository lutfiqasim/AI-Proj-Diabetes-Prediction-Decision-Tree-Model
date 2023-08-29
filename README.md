# Diabetes Prediction Decision Tree Model
This project implements a decision tree-based model to predict diabetes using health-related features. The Python script processes diabetes-related data, creates and trains decision tree classifiers, evaluates their accuracy, and visualizes decision tree graphs. It provides a user-friendly command-line interface for various interactions.

## Features

- Data processing and splitting into training and test sets.
- Model training using scikit-learn's decision tree classifier.
- Model evaluation and accuracy calculation.
- Visualization of decision tree graphs.
- User-friendly command-line interface.

## Getting Started

1. Clone the repository: `git clone https://github.com/your-username/diabetes-prediction.git`
2. Navigate to the project directory: `cd diabetes-prediction`
3. Install required dependencies: `pip install graphviz matplotlib scikit-learn prettytable`

## Usage

Run the script using the command: `python DecisionTree.py dataset_file.csv "Diabetic"`

Options:
- View dataset statistics
- Plot class distribution
- Create and manage models
- Test model accuracy
- Visualize decision tree graphs
- Delete models

## Dataset
The dataset should be in CSV format with relevant attributes:
- NPG: Number of times pregnant
- PGL: Plasma glucose concentration
- DIA: Diastolic blood pressure
- TSF: Triceps skin fold thickness
- INS: 2-Hour serum insulin
- BMI: Body mass index
- DPF: Diabetes pedigree function
- AGE: Age
- Diabetic: Class variable (0: Non-Diabetic, 1: Diabetic)


## Acknowledgments

This project uses the [scikit-learn](https://scikit-learn.org) library for machine learning tasks and [PrettyTable](https://pypi.org/project/prettytable/) for tabular data formatting.

Feel free to contribute to this project by opening issues or pull requests!
