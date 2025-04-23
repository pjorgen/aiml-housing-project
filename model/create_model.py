import argparse
import json
import pathlib
import pickle
from typing import List, Tuple
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data relative to this file
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics relative to this file

# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION_BASIC = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

# More comprehensive list of columns that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'waterfront', 'view', 'condition', 'grade', 'lat', 'long', 'yr_built', 'yr_renovated',
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode', 'sqft_living15', 'sqft_lot15'
]


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containing with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(demographics_path,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def train_model_with_max_attempts(x, y, max_attempts=5):
    """
    Create the model using a sci-kit learning pipeline
    We want to ensure the best possible fit, so we iterate up to max_attempts
    Additionally, we switch from KNeighborsRegressor to RandomForestRegressor which is a better fit for this sales data
    """
    attempt = 0
    best_score = 0.0
    best_model = None

    while attempt < max_attempts:
        print("Training attempt {} of {}".format(attempt + 1, max_attempts))
        model = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', RandomForestRegressor())
        ])

        # Evaluate the model for performance
        scores = cross_val_score(model, x, y, cv=5, scoring='r2')
        avg_score = np.mean(scores)
        print("Average R2 score: {} for attempt {}".format(avg_score, attempt + 1))

        # Check score against best so far
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
        attempt += 1

    # Fit the best model and return it
    best_model.fit(x, y)
    return best_model


def main():
    """Parse arguments, load data, train model, and export artifacts."""

    # Parse arguments
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--model-name", type=str, required=True, help='Name of the output model. Used for output directory.')
    parser.add_argument("--basic", type=bool, default=False, required=False, help='Use basic sales parameters.')
    parser.add_argument("--max-attempts", type=int, default=5, required=False, help='Maximum number of attempts to find a model.')
    args = parser.parse_args()

    if args.basic:
        columns = SALES_COLUMN_SELECTION_BASIC
    else:
        columns = SALES_COLUMN_SELECTION

    # Load data. Use a set random state so we can deterministically test later
    print("Loading data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, columns)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    # Train the model to a minimum threshold using just the training data
    print("Training model...")
    model = train_model_with_max_attempts(x_train, y_train, args.max_attempts)

    current_dir = pathlib.Path(__file__).resolve().parent
    output_dir = pathlib.Path(current_dir.parent / 'app' / 'models' / args.model_name)
    print("Output directory: {}".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    print("Exporting artifacts...")
    with open(output_dir / "model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(output_dir / "model_features.json", 'w') as f:
        json.dump(list(x_train.columns), f)


if __name__ == "__main__":
    main()
