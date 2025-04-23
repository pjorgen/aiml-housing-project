import argparse
import pathlib
import pandas as pd
import math
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json


HOUSE_DATA_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"


def load_data():
    # Load the data
    house_data = pd.read_csv(HOUSE_DATA_PATH, dtype={'zipcode': str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
    return house_data, demographics


def merge_data(house_data, demographics):
    # Merge the house data with demographics
    data = house_data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    return data


def load_and_merge_data():
    house_data, demographics = load_data()
    return merge_data(house_data, demographics)


def calculate_and_print_metrics(y_test, y_hat, model_name):
    # Calculate metrics
    mse = mean_squared_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    mae = mean_absolute_error(y_test, y_hat)
    rmse = math.sqrt(mse)
    bias = (y_hat - y_test).mean()

    # Print results
    print(f"Evaluation Results for model {model_name}:")
    print(f"Mean Squared Error (MSE):       {mse}")
    print(f"R-squared (R²):                 {r2}")
    print(f"Mean Absolute Error (MAE):      {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Bias (Average Prediction):      {bias}")


def main():
    """Evaluate the model."""

    # Parse arguments
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--model-name", type=str, required=True, help='Name of the model. Used to determine model path')
    args = parser.parse_args()

    # Load and merge data
    data = load_and_merge_data()

    # Set the model path and load features
    current_dir = pathlib.Path(__file__).resolve().parent
    model_path = pathlib.Path(current_dir.parent / 'app' / 'models' / args.model_name)
    features_path = pathlib.Path(model_path / 'model_features.json')
    with open(features_path, 'r') as f:
        model_features = json.load(f)

    # Split out training data using the same random state as training
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        data[model_features], data['price'], random_state=42)

    # Load the model
    with open(model_path / "model.pkl", 'rb') as f:
        model = pickle.load(f)

    # Make predictions
    y_hat = model.predict(x_test)

    # Print results
    calculate_and_print_metrics(y_test, y_hat, args.model_name)


if __name__ == "__main__":
    main()
