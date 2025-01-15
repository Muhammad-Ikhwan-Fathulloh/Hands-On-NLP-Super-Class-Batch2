from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import string
import unicodedata
import os

# Define FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti "*" dengan domain spesifik untuk keamanan lebih baik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model configurations
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Dynamically load all categories from the data/names directory
data_path = 'data/names'
all_categories = [os.path.splitext(filename)[0] for filename in os.listdir(data_path) if filename.endswith('.txt')]
n_categories = len(all_categories)

# Define model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = torch.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Load trained model
rnn = RNN(n_letters, 128, n_categories)  # Use the hidden_size from training
rnn.load_state_dict(torch.load("rnn.pt"))
rnn.eval()

# Utility functions
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i]

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

# Input and output models for API
class PredictionRequest(BaseModel):
    name: str

class PredictionResponse(BaseModel):
    category: str
    confidence: float

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_name = unicodeToAscii(request.name)
    input_tensor = lineToTensor(input_name)
    with torch.no_grad():
        output = evaluate(input_tensor)
        category = categoryFromOutput(output)
        confidence = torch.exp(output.max()).item()
    return PredictionResponse(category=category, confidence=confidence)

@app.get("/")
async def root():
    return {"message": "RNN Model is success running"}