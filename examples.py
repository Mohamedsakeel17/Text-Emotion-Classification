"""Run the classifier on a few example sentences and save one chart each."""
from emotion_classifier import EmotionClassifier
from visualize import plot_emotions

SENTENCES = [
    "I just got the job offer, I can't stop smiling!",
    "I can't believe they cancelled my flight again.",
    "This room is so dark, I think someone is following me.",
    "The meeting is at 3 pm in room 204.",
]

if __name__ == "__main__":
    clf = EmotionClassifier()
    for i, s in enumerate(SENTENCES, 1):
        scores = clf.predict(s)
        top = next(iter(scores))
        print(f"{s}\n  -> {top} ({scores[top]:.2f})")
        plot_emotions(scores, s, save_path=f"example_{i}.png", show=False)
