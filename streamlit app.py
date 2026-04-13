import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

# Load Model + Config

@st.cache_resource
def load_model():
    SAVE_DIR = "package/model_v1"

    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_DIR)

    with open(os.path.join(SAVE_DIR, "app_config.json")) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, config, device


model, tokenizer, config, device = load_model()

# UI

st.set_page_config(page_title=" Medical Translator")

st.title(" Domain-Specific Translator (EN → KO)")
st.markdown(" Research prototype — not for clinical use")

text = st.text_area("Enter English text:", height=150)

# Inference

def translate(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config["max_input_length"]
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config["max_output_length"],
            min_length=config["min_length"],
            num_beams=config["num_beams"],
            no_repeat_ngram_size=config["no_repeat_ngram_size"],
            early_stopping=config["early_stopping"]
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if st.button("Translate"):

    if not text.strip():
        st.warning("Please enter text")
    else:
        output = translate(text)

        st.subheader(" Translation")
        st.write(output)

        # Debug insights (very strong signal)
        st.subheader(" Diagnostics")
        st.write(f"Input Length: {len(text.split())}")
        st.write(f"Output Length: {len(output.split())}")


# UX Upgrade

if len(output.strip()) < 5:
    st.error(" Low-confidence / degenerate output detected")
