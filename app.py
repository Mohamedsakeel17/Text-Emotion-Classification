"""Real-time text emotion classifier (command line).

Usage:
    python app.py                       # interactive: type a sentence, get the emotions
    python app.py --text "I love this"  # classify one sentence
    python app.py --text "..." --save out.png --no-show
"""
import argparse

from emotion_classifier import EmotionClassifier
from visualize import plot_emotions


def show_result(scores, text, save, show):
    print("\n".join(f"  {k:<9} {v:.3f}" for k, v in scores.items()))
    plot_emotions(scores, text, save_path=save, show=show)


def main():
    parser = argparse.ArgumentParser(description="Text emotion classification (DistilRoBERTa)")
    parser.add_argument("--text", help="classify a single sentence and exit")
    parser.add_argument("--save", help="save the chart to this image path")
    parser.add_argument("--no-show", action="store_true", help="do not open the chart window")
    args = parser.parse_args()

    clf = EmotionClassifier()
    show = not args.no_show

    if args.text:
        show_result(clf.predict(args.text), args.text, args.save, show)
        return

    print("Type a sentence and press Enter (empty line to quit).")
    while True:
        text = input("\n> ").strip()
        if not text:
            break
        show_result(clf.predict(text), text, args.save, show)


if __name__ == "__main__":
    main()
