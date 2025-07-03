import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# File paths
RAW_DATA_PATH = "data/raw_data.json"
TRAIN_CSV_PATH = "data/train.csv"
VAL_CSV_PATH = "data/val.csv"

def load_and_process_data(raw_path):
    """
    Load the original sarcasm JSON dataset and convert it to a DataFrame.
    Keeps only the 'headline' and 'is_sarcastic' fields.
    Drops duplicates and missing values.
    """
    records = []
    with open(raw_path, "r") as f:
        buffer = ""
        for line in f:
            buffer += line.strip()
            if buffer.endswith("}"):
                try:
                    obj = json.loads(buffer)
                    records.append(obj)
                    buffer = ""
                except json.JSONDecodeError:
                    continue  # 如果有脏数据就跳过

    df = pd.DataFrame.from_records(records)
    df = df[["headline", "is_sarcastic"]]
    df.rename(columns={"is_sarcastic": "label"}, inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def split_and_save(df, train_path, val_path, test_size=0.2, random_state=42):
    """
    Split the full DataFrame into training and validation sets,
    then save them as CSV files.
    Stratify by label to preserve class balance.
    """
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved {len(train_df)} training samples to {train_path}")
    print(f"Saved {len(val_df)} validation samples to {val_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = load_and_process_data(RAW_DATA_PATH)
    split_and_save(df, TRAIN_CSV_PATH, VAL_CSV_PATH)
