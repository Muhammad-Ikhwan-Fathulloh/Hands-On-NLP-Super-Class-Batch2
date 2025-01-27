from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

# Define the FastAPI app
app = FastAPI(title="Spam Classifier API", description="A BERT-based spam classification API", version="1.0")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Define the BERT-based model architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the model and set it to evaluation mode
model = BERT_Arch(bert)
model.load_state_dict(torch.load("spam_model.pt", map_location=device))
model = model.to(device)
model.eval()

# Define a request schema
class SpamPredictionRequest(BaseModel):
    text: str

# Define a response schema
class SpamPredictionResponse(BaseModel):
    label: str
    confidence: float

# Define a prediction function
def predict_spam(text: str):
    # Tokenize input text
    tokens = tokenizer.batch_encode_plus(
        [text],
        max_length=25,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    # Convert tokens to tensors
    input_ids = torch.tensor(tokens['input_ids']).to(device)
    attention_mask = torch.tensor(tokens['attention_mask']).to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_label = torch.max(probs, dim=1)

    # Map labels to human-readable categories
    label_map = {0: "Not Spam", 1: "Spam"}
    return label_map[predicted_label.item()], confidence.item()

# Define API endpoint for prediction
@app.post("/predict", response_model=SpamPredictionResponse)
async def predict(request: SpamPredictionRequest):
    try:
        label, confidence = predict_spam(request.text)
        return SpamPredictionResponse(label=label, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "BERT Model is success running"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
