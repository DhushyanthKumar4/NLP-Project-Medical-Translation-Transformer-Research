# IMPORTS

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from fastapi import FastAPI
from pydantic import BaseModel

# CONFIG

CONFIG = {
    "model_name": "t5-small",
    "task": "en_to_ko_translation",
    "max_input_length": 128,
    "max_output_length": 80,
    "num_beams": 4,
    "min_length": 10,
    "no_repeat_ngram_size": 2,
    "early_stopping": True,
    "device": "auto",
    "version": "v1.0"
}

# Point to one of the models saved in Experiment A (e.g., the 100% data model)
MODEL_PATH = "./exp_1.0"
SAVE_DIR = "./package/model_v1"

# SAVE MODEL + TOKENIZER

def save_model():
    # Only load from MODEL_PATH if it's not the SAVE_DIR itself to avoid recursive loading issues
    if os.path.exists(MODEL_PATH) and MODEL_PATH != SAVE_DIR:
        model_to_save = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        tokenizer_to_save = AutoTokenizer.from_pretrained(MODEL_PATH)
    else:
        # If MODEL_PATH is not valid or already pointing to SAVE_DIR, use a fresh model
        print("Warning: MODEL_PATH not found or is SAVE_DIR, loading fresh t5-small for packaging.")
        model_to_save = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        tokenizer_to_save = AutoTokenizer.from_pretrained("t5-small")

    os.makedirs(SAVE_DIR, exist_ok=True)

    model_to_save.save_pretrained(SAVE_DIR)
    tokenizer_to_save.save_pretrained(SAVE_DIR)

    # Save custom CONFIG to a separate file to avoid overwriting model's essential config.json
    with open(os.path.join(SAVE_DIR, "app_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Model and custom app config saved to {SAVE_DIR}")

# Ensure the model is saved before attempting to load it
# This makes the cell runnable on its own for the first time setup
save_model()

# LOAD MODEL

def load_model():
    # Load custom app config
    with open(os.path.join(SAVE_DIR, "app_config.json")) as f:
        app_config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(SAVE_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return model, tokenizer, app_config, device # Return app_config instead of CONFIG

model, tokenizer, config, device = load_model()

# INFERENCE FUNCTION

def translate(text: str):
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

#  CLI MODE

def run_cli():
    while True:
        text = input("\nEnter English sentence (or 'exit'): ")
        if text.lower() == "exit":
            break
        print("Translation:", translate(text))

# FASTAPI APP

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/translate")
def translate_api(input: InputText):
    return {"translation": translate(input.text)}

# ENTRY POINT

if __name__ == "__main__":
    print("\nRunning in script mode. Choose an option:")
    mode = input("[1] Run CLI  [2] Nothing (only setup models) : ")

    if mode == "1":
        run_cli()
    else:
        print("Exiting script mode. Model is packaged and ready.")
