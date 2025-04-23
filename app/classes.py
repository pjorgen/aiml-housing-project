from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar('T')

class BasicFeatureSet(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: int

class EnhancedFeatureSet(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class PredictionRequest(BaseModel, Generic[T]):
    model: str
    features: T

class BasicPredictionRequest(PredictionRequest[BasicFeatureSet]):
    pass

class EnhancedPredictionRequest(PredictionRequest[EnhancedFeatureSet]):
    pass

class ResponseMetadata(BaseModel):
    model: str
    request_id: str
    timestamp: str

class PredictionResponse(BaseModel):
    predicted_price: float
    metadata: ResponseMetadata
