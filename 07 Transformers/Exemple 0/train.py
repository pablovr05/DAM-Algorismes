import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    label_encoders = {}
    for column in ['brand', 'color', 'cpu', 'ram', 'OS', 'special_features', 'graphics', 'graphics_coprocessor']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
    X = df.drop(columns=['price', 'model'])
    y = df['price']
    return train_test_split(X, y, test_size=0.2, random_state=42)

class ComputerPriceDataset(Dataset):
    def __init__(self, features, prices, tokenizer, max_length=64):
        self.features = features.values
        self.prices = prices.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = ' '.join(map(str, self.features[idx]))
        label = str(self.prices[idx])
        
        inputs = self.tokenizer(
            features, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        labels = self.tokenizer(
            label, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        ).input_ids
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }
class ComputerPriceDataset(Dataset):
    def __init__(self, features, prices, tokenizer, max_length=64):
        self.features = features.values
        self.prices = prices.values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = ' '.join(map(str, self.features[idx]))
        label = str(self.prices[idx])
        
        inputs = self.tokenizer(
            features, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        )
        
        labels = self.tokenizer(
            label, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        ).input_ids
        
        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

def train():
    X_train, X_val, y_train, y_val = preprocess_data("./data/laptops.csv")
    model_name = "t5-small"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    train_dataset = ComputerPriceDataset(X_train, y_train, tokenizer)
    val_dataset = ComputerPriceDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" or device.type == "mps":
        print(f"Using device: {device} (GPU accelerated)")
    else:
        print(f"Using device: {device} (CPU based)")
        
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} completed")

    torch.save(model.state_dict(), './trained_model.pth')

if __name__ == "__main__":
    train()
