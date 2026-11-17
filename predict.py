"""Command-line next-digit prediction."""

import argparse

import torch

from model import NextDigitModel


def main():
    parser = argparse.ArgumentParser(description="Predict the next digit in a sequence.")
    parser.add_argument("digits", nargs="+", type=int, help="Input digits (0-9)")
    args = parser.parse_args()
    if len(args.digits) < 3:
        parser.error("provide at least three input digits")
    if any(digit < 0 or digit > 9 for digit in args.digits):
        parser.error("each digit must be between 0 and 9")

    model = NextDigitModel()
    try:
        state = torch.load("model.pt", map_location="cpu", weights_only=True)
    except FileNotFoundError:
        parser.error("model.pt not found; run train.py first")
    model.load_state_dict(state)
    model.eval()

    inputs = torch.tensor([args.digits], dtype=torch.long)
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)[0]
    top_probabilities, top_digits = probabilities.topk(3)
    print("Predicted next digit: {}".format(int(top_digits[0])))
    print("Top 3:")
    for digit, confidence in zip(top_digits.tolist(), top_probabilities.tolist()):
        print("  {}: {:.2%}".format(digit, confidence))


if __name__ == "__main__":
    main()
