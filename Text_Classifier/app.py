# Save this code as `main.py`
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load the trained model
model = joblib.load('intent_classification.joblib')

# Initialize FastAPI
app = FastAPI()

# Define the input data structure
class TextInput(BaseModel):
    text: str

# Define the prediction endpoint
@app.post("/predict/")
async def predict_intent(input: TextInput):
    try:
        # Preprocess input text (if needed, e.g., lowercase)
        processed_text = input.text.lower()
        
        # Predict intent
        prediction = model.predict([processed_text])
        
        # Return the prediction as a JSON response
        return {"intent": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Define a health check endpoint
@app.get("/")
async def root():
    return {"message": "Intent Classification Model is success running"}

@app.get("/profile")
async def profile():
    return {"name": "Ikhwan", "description": "Learn NLP"}