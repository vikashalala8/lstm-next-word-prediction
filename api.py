from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_next_word


# Create FastAPI application
app = FastAPI()


# Request structure
class TextInput(BaseModel):

    text: str


# Home route
@app.get("/")
def home():

    return {
        "message": "LSTM Next Word Prediction API"
    }


# Prediction route
@app.post("/predict")
def predict(data: TextInput):

    next_word = predict_next_word(
        data.text
    )

    return {
        "input": data.text,
        "predicted_word": next_word
    }