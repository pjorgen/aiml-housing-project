import json
import pathlib
import requests
import pandas as pd


current_dir = pathlib.Path(__file__).resolve().parent
INPUTS_PATH = pathlib.Path(current_dir.parent / "model" / "data" / "future_unseen_examples.csv")
DEMOGRAPHICS_PATH = pathlib.Path(current_dir.parent / "model" / "data" / "kc_house_data.csv")

# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION_BASIC = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

# More comprehensive list of columns that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
    'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15'
]

URL = "http://localhost:5000/"
HEADERS = {"Content-Type": "application/json"}


def load_test_cases(columns):
    # Load test data from future unseen examples
    df = pd.read_csv(INPUTS_PATH)
    df_filtered = df[columns]
    data = df_filtered.to_dict(orient='records')
    return data


def test_index():
    response = requests.get(URL)
    assert response.status_code == 200

    parsed = response.json()
    pretty = json.dumps(parsed, indent=4)
    print(f"Response: {pretty}")


def test_models():
    response = requests.get(URL + "/models")
    assert response.status_code == 200

    parsed = response.json()
    pretty = json.dumps(parsed, indent=4)
    print(f"Response: {pretty}")


def test_iterator(model_name, columns, url):
    # Load test data using basic column set
    data = load_test_cases(columns)

    # Loop over data to create payload
    for row in data:
        payload = json.dumps({"model": model_name, "features": row})
        response = requests.post(URL + url, data=payload, headers=HEADERS)
        assert response.status_code == 200

        parsed = response.json()
        pretty = json.dumps(parsed, indent=4)
        print(f"Response: {pretty}")


def main():
    # Test index
    test_index()

    # Test the basic model
    test_iterator("v1-basic", SALES_COLUMN_SELECTION_BASIC, "/predict-basic")

    # Test the comprehensive model
    test_iterator("v1", SALES_COLUMN_SELECTION, "/predict")


if __name__ == "__main__":
    main()
