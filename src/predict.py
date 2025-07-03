import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Load the saved model and tokenizer
MODEL_DIR = "outputs/sarcasm_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

def predict(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()

    # Output result
    label = "Sarcastic 🤨" if prediction == 1 else "Not Sarcastic 🙂"
    confidence = probs[0][prediction].item()
    return label, confidence

# CLI loop
if __name__ == "__main__":
    print("Enter a sentence to check for sarcasm (type 'q' to quit):")
    while True:
        text = input("\n> ")
        if text.lower() in {"q", "quit"}:
            break
        label, confidence = predict(text)
        print(f"→ Prediction: {label} (Confidence: {confidence:.2f})")
