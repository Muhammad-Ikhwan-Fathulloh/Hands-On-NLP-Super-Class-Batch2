import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

class TransformerClassifier(torch.nn.Module):
    def __init__(self, num_labels, hidden_dim=768, nhead=8, num_layers=6, max_length=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(30522, hidden_dim)  # Using BERT vocab size
        self.positional_encoding = torch.nn.Parameter(torch.zeros(max_length, hidden_dim))
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids) + self.positional_encoding[:input_ids.size(1), :]
        embeddings = self.dropout(embeddings)
        transformer_out = self.transformer(embeddings.permute(1, 0, 2), src_key_padding_mask=(attention_mask == 0))
        pooled_output = transformer_out.mean(dim=0)  # Pooling
        logits = self.classifier(pooled_output)
        return logits

# Load trained model and tokenizer
num_labels = 3  # Adjust according to your dataset
model = TransformerClassifier(num_labels=num_labels)
model.load_state_dict(torch.load("intent_transformer_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")

# Label map
label_map = {
    0: "greeting",
    1: "sekarang_jam_berapa",
    2: "siapa_anda",  # Update with actual labels
}

# FastAPI app setup
app = FastAPI(title="Intent Classification API", version="1.0")

class TextRequest(BaseModel):
    text: str

class BatchRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    intent: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    text = request.text
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    # Tokenize input text
    tokens = tokenizer(
        text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = softmax(logits, dim=1)
        confidence, predicted_label = torch.max(probabilities, dim=1)

    # Return prediction
    intent = label_map[predicted_label.item()]
    return {"intent": intent, "confidence": confidence.item()}

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchRequest):
    texts = request.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty.")
    
    predictions = []
    for text in texts:
        # Tokenize input text
        tokens = tokenizer(
            text, padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Make prediction
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = softmax(logits, dim=1)
            confidence, predicted_label = torch.max(probabilities, dim=1)

        # Append result
        intent = label_map[predicted_label.item()]
        predictions.append({"intent": intent, "confidence": confidence.item()})
    
    return predictions

# Health check
@app.get("/")
async def health_check():
    return {"message": "Intent Classification API is up and running!"}
