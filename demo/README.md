---
title: Tweet Sentiment Analyzer
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
short_description: DistilBERT sentiment classifier trained on Sentiment140
---

# Tweet Sentiment Analyzer

Type a sentence and see whether the model reads it as positive or negative, with its confidence.

The model is DistilBERT fine-tuned on 400,000 tweets from Sentiment140 on a laptop CPU. Training code, evaluation and the REST API are at [github.com/ngugiauraa/sentiment-analyzer](https://github.com/ngugiauraa/sentiment-analyzer).
