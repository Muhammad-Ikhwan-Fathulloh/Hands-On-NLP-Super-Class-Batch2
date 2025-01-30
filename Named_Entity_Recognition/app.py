from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
from torch import nn
from transformers import BertTokenizerFast, BertForTokenClassification

# FastAPI app initialization
app = FastAPI()

# Define model and labels
class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(ids_to_labels))

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=False)
        return outputs

# Pre-defined label mappings
ids_to_labels = {
    0: 'B-art', 1: 'B-eve', 2: 'B-geo', 3: 'B-gpe', 4: 'B-nat',
    5: 'B-org', 6: 'B-per', 7: 'B-tim', 8: 'I-art', 9: 'I-eve',
    10: 'I-geo', 11: 'I-gpe', 12: 'I-nat', 13: 'I-org', 14: 'I-per',
    15: 'I-tim', 16: 'O'
}

# Initialize tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertModel()
model.load_state_dict(torch.load("ner-model.pth", map_location=torch.device('cpu')))
model.eval()

# Pydantic schemas for request and response
class TextInput(BaseModel):
    text: str

class EntityPrediction(BaseModel):
    word: str
    entity: str

class PredictionOutput(BaseModel):
    entities: List[EntityPrediction]

# Utility function for entity prediction
def predict_entities(sentence: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenized = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)[0]
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    word_ids = tokenized.word_ids()
    entities = []
    previous_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != previous_word_idx:  # Skip subwords and padding
            label_id = predictions[idx]
            if ids_to_labels[label_id] != "O":  # Skip non-entity tokens
                entities.append({
                    "word": tokenizer.convert_ids_to_tokens(input_ids[0][idx]),
                    "entity": ids_to_labels[label_id]
                })
        previous_word_idx = word_idx
    return entities

# FastAPI endpoints
@app.post("/predict", response_model=PredictionOutput)
def predict(text_input: TextInput):
    try:
        entities = predict_entities(text_input.text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
