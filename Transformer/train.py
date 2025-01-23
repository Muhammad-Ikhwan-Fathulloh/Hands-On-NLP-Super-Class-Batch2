import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer
import pandas as pd
import string

# Load dataset
data = pd.read_csv("data3.csv")

# Preprocessing function
def preprocess_text(text):
    # Lowercasing and removing punctuation
    return text.lower().translate(str.maketrans('', '', string.punctuation))

data['text'] = data['text'].apply(preprocess_text)

# Split data into features and labels
X = data['text'].tolist()
y = data['intent'].astype('category')
label_map = {label: idx for idx, label in enumerate(y.cat.categories)}
y = y.map(label_map).tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")

# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = IntentDataset(X_train, y_train, tokenizer)
test_dataset = IntentDataset(X_test, y_test, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Transformer model for classification
class TransformerClassifier(nn.Module):
    def __init__(self, num_labels, hidden_dim=768, nhead=8, num_layers=6, max_length=128, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(30522, hidden_dim)  # Using BERT vocab size
        self.positional_encoding = nn.Parameter(torch.zeros(max_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids) + self.positional_encoding[:input_ids.size(1), :]
        embeddings = self.dropout(embeddings)
        transformer_out = self.transformer(embeddings.permute(1, 0, 2), src_key_padding_mask=(attention_mask == 0))
        pooled_output = transformer_out.mean(dim=0)  # Pooling
        logits = self.classifier(pooled_output)
        return logits

# Initialize model, optimizer, and loss function
num_labels = len(label_map)
model = TransformerClassifier(num_labels=num_labels)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Set device
device = "cpu"
model.to(device)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions)

# Metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=list(label_map.keys())))

# Save the model
torch.save(model.state_dict(), "intent_transformer_model.pth")
