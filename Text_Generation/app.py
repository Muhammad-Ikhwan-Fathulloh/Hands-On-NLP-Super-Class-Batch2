from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Set model directory
model_dir = "model"

# Initialize the device
device = torch.device("cpu")

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# FastAPI app
app = FastAPI(title="Text Generation Model Serving")

# Request body schema
class TextGenerationRequest(BaseModel):
    input_text: str
    num_return_sequences: int = 256

# Response body schema
class TextGenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=TextGenerationResponse)
def generate_text(request: TextGenerationRequest):
    """
    Generate text based on the input prompt.
    """
    try:
        # Generate text
        results = pipe(request.input_text, num_return_sequences=request.num_return_sequences)
        # Return the first generated sequence
        return TextGenerationResponse(generated_text=results[0]["generated_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Generation API!"}
