"""
TakeMeter — NBA Post Classifier
Gradio interface for the fine-tuned DistilBERT model.

Setup:
    pip install gradio transformers torch

Usage:
    python app.py

Then open http://localhost:7860 in your browser.

Note: Requires the saved model weights in ./nba_classifier/
Download nba_classifier.zip from the Colab notebook outputs
and unzip it in the repo root before running.
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./nba_classifier"
ID2LABEL = {0: "analysis", 1: "hot_take", 2: "reaction"}

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")


def classify_post(text):
    if not text.strip():
        return "Please enter a post.", {}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    pred_label = ID2LABEL[pred_id]
    confidence_dict = {ID2LABEL[i]: float(probs[i]) for i in range(len(ID2LABEL))}

    return f"**{pred_label}** ({confidence_dict[pred_label]:.1%} confident)", confidence_dict


demo = gr.Interface(
    fn=classify_post,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Paste an r/nba post or comment here...",
        label="NBA Post"
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="TakeMeter — NBA Post Classifier",
    description="Classifies r/nba posts as **analysis**, **hot_take**, or **reaction** using a fine-tuned DistilBERT model.",
    examples=[
        ["Jokic's assist-to-turnover ratio in playoff series he's won versus lost is 5.2 vs 3.1. The difference isn't scoring — it's ball security."],
        ["LeBron is the greatest player ever and nobody who argues otherwise is watching the same sport."],
        ["LETS GOOOOOOO. I knew it. I knew it. I said this exact thing in this subreddit three weeks ago."],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()