import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["USE_TF_KERAS"] = "1"

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = TFBertForSequenceClassification.from_pretrained("outputs/sarcasm_model/")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model()

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=100)
    outputs = model(inputs)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    pred = np.argmax(probs)
    return pred, probs

# UI
st.title("Sheldon VS Sarcasm")

text = st.text_input("Enter a sentence:")

if not text:
    st.image("assets/sheldon_default.jpg", use_container_width=True)
    st.markdown("<h4 style='text-align: center;'>CAUSE I CAN TELL!</h4>", unsafe_allow_html=True)
else:
    pred, probs = predict(text)
    confidence = probs[pred]

    if pred == 1:
        st.success(f"Sarcastic (Confidence: {confidence:.2f})")
        st.image("assets/sarcasm_yes.jpg", use_container_width=True)
        st.markdown("<h4 style='text-align: center;'>Definitely sarcastic.</h4>", unsafe_allow_html=True)
    else:
        st.info(f"Not Sarcastic (Confidence: {confidence:.2f})")
        st.image("assets/sheldon_dancing.gif", use_container_width=True)
        st.markdown("<h4 style='text-align: center;'>Seems sincere!<br>Enjoy this flawless execution of celebratory movement.</h4>", unsafe_allow_html=True)