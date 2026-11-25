"""Text emotion classification with a pre-trained DistilRoBERTa model.

Model: j-hartmann/emotion-english-distilroberta-base (Hugging Face),
a DistilRoBERTa model fine-tuned to predict 7 emotions:
anger, disgust, fear, joy, neutral, sadness, surprise.
"""
from transformers import pipeline

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


class EmotionClassifier:
    def __init__(self, model_name: str = MODEL_NAME):
        # top_k=None returns the score for every emotion, not just the best one
        self.pipe = pipeline("text-classification", model=model_name, top_k=None)

    def predict(self, text: str) -> dict:
        """Return {emotion: probability} for one piece of text, highest first."""
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")
        results = self.pipe(text, truncation=True)[0]
        scores = {r["label"]: float(r["score"]) for r in results}
        return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))

    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]
