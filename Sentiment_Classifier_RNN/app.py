from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from torch.nn import functional as F
from torchtext import data
import spacy
import os

# Initialize FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# Load the tokenizer
nlp = spacy.load('en_core_web_sm')

# Model and Vocabulary Loading
class SentimentRNN(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = torch.nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to("cpu"))
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

# Define Request Model
class SentimentRequest(BaseModel):
    sentence: str

# Load model
def load_model():
    INPUT_DIM = 25002  # Change based on your training setup
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = 1  # Replace with `TEXT.vocab.stoi[TEXT.pad_token]` used during training
    
    model = SentimentRNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
    model.load_state_dict(torch.load("sentiment-model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

# Initialize model
model = load_model()

# Define prediction function
def predict_sentiment(sentence: str):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi.get(t, TEXT.vocab.stoi['<unk>']) for t in tokenized]  # Handle unknown words
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

# API Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}

@app.post("/predict/")
def analyze_sentiment(request: SentimentRequest):
    try:
        sentiment_score = predict_sentiment(request.sentence)
        sentiment = "positive" if sentiment_score >= 0.5 else "negative"
        return {"sentence": request.sentence, "sentiment": sentiment, "score": sentiment_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred during prediction: {e}")