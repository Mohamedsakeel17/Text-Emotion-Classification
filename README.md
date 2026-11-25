# Text Emotion Classification Using Transformers

A real-time text emotion classifier built on **DistilRoBERTa**, with Matplotlib charts of the emotion scores so the model's output is easy to interpret.

> Re-implementation (2026) of an earlier coursework project.

## What it does
- Takes a sentence and predicts probabilities for 7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise.
- Uses the pre-trained Hugging Face model [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), a DistilRoBERTa model already fine-tuned for emotion classification. This project uses the model for inference and does not train it.
- Runs interactively: type a sentence, see the scores and a bar chart straight away.
- Saves the chart as an image if you ask for it.

## Setup
```bash
git clone https://github.com/Mohamedsakeel17/text-emotion-classification.git
cd text-emotion-classification
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
The model (about 330 MB) downloads automatically on first run, so you need an internet connection once.

## Usage
```bash
python app.py                                   # interactive mode
python app.py --text "I can't stop smiling!"    # one sentence
python app.py --text "..." --save chart.png --no-show
python examples.py                              # runs 4 examples, saves example_1.png ... example_4.png
```

## Files
| File | Purpose |
|---|---|
| `emotion_classifier.py` | Loads the model and returns emotion probabilities |
| `visualize.py` | Matplotlib bar chart of the scores |
| `app.py` | Command-line, real-time interface |
| `examples.py` | Example sentences and saved charts |

## Tech stack
Python, Hugging Face Transformers, PyTorch, Matplotlib

## Limitations
- English text only.
- Predicts one of 7 basic emotions, so mixed or subtle emotions and sarcasm can be misclassified.
- Scores are model probabilities, not ground truth.

## Author
Mohamed Sakeel A
