import json
import pathlib
import pickle
import classes
from typing import TypeVar
import pandas as pd
import uuid


T = TypeVar('T')
DEMOGRAPHICS_DATA = pd.read_csv("data/zipcode_demographics.csv")


def load_model(model_name: str):
    model_pkl = pathlib.Path( "models" , model_name , "model.pkl")
    feature_file = pathlib.Path( "models" , model_name , "model_features.json")
    with open(model_pkl, 'rb') as f:
        model = pickle.load(f)
    with open(feature_file, 'r') as f:
        model_features = json.load(f)
    return model, model_features


def get_prediction(request: classes.PredictionRequest[T]):
    # Load the model and features
    model, model_features = load_model(request.model)

    # Merge request features with demographic data
    data = pd.DataFrame([request.features.dict()])
    data = data.merge(DEMOGRAPHICS_DATA, on="zipcode", how="left")

    # Re-order features to match training
    data = data[model_features]

    # Get prediction
    prediction = model.predict(data)

    # Create response
    metadata = classes.ResponseMetadata(
        model = request.model,
        request_id = str(uuid.uuid4()),
        timestamp = str(pd.Timestamp.now())
    )
    response = classes.PredictionResponse(
        predicted_price = prediction,
        metadata = metadata
    )
    return response