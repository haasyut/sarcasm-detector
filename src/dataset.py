import torch
from torch.utils.data import Dataset
import pandas as pd

class SarcasmDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=64):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headline = str(self.data.iloc[idx]["headline"])
        label = int(self.data.iloc[idx]["label"])

        encoding = self.tokenizer(
            headline,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long)
        )
