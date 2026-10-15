"""
Public demo on Hugging Face Spaces: type a sentence, get the sentiment.
Runs the same model as the API, downloaded from the Hugging Face Hub.
"""

import gradio as gr
from transformers import pipeline

MODEL_ID = "ngugiauraa/distilbert-sentiment140"
classifier = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=48)

EXAMPLES = [
    "Just got my results and I passed everything!",
    "Stuck in Nairobi traffic for 3 hours, again",
    "The new update finally fixed the battery drain",
    "Customer care hung up on me twice today",
    "Not bad at all for the price",
]


def classify(text: str):
    if not text.strip():
        return {}
    scores = classifier(text)[0]
    return {s["label"].lower(): round(s["score"], 4) for s in scores}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Text", placeholder="Type a tweet or short review...", lines=2),
    outputs=gr.Label(label="Sentiment", num_top_classes=2),
    examples=EXAMPLES,
    title="Tweet Sentiment Analyzer",
    description=(
        "DistilBERT fine-tuned on 400,000 tweets from Sentiment140, trained on a laptop CPU. "
        "Source and evaluation: github.com/ngugiauraa/sentiment-analyzer"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
