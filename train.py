"""Train, evaluate, and save the next-digit model."""

import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import PATTERN_TYPES, generate_dataset
from model import NextDigitModel, most_frequent_baseline, repeat_last_baseline


SEED = 42
BATCH_SIZE = 128
EPOCHS = 15
PATIENCE = 4


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def evaluate(model, inputs, labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            batch = inputs[start : start + BATCH_SIZE]
            predictions.append(model(batch).argmax(dim=1))
    predictions = torch.cat(predictions)
    return predictions, (predictions == labels).float().mean().item()


def main():
    set_seeds(SEED)
    train_x, train_y, _ = generate_dataset(50_000, seed=SEED)
    val_x, val_y, _ = generate_dataset(5_000, seed=SEED + 1)
    test_x, test_y, test_tags = generate_dataset(5_000, seed=SEED + 2)

    train_inputs = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y)
    val_inputs = torch.from_numpy(val_x)
    val_labels = torch.from_numpy(val_y)
    test_inputs = torch.from_numpy(test_x)
    test_labels = torch.from_numpy(test_y)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = NextDigitModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses = []
    val_accuracies = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)

        train_loss = total_loss / len(train_inputs)
        _, val_accuracy = evaluate(model, val_inputs, val_labels)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        print(
            "Epoch {}/{} - train loss: {:.4f} - val accuracy: {:.2%}".format(
                epoch, EPOCHS, train_loss, val_accuracy
            ),
            flush=True,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping at epoch {}.".format(epoch), flush=True)
                break

    model.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    test_predictions, test_accuracy = evaluate(model, test_inputs, test_labels)
    repeat_predictions = repeat_last_baseline(test_inputs)
    frequent_predictions = most_frequent_baseline(test_inputs)
    repeat_accuracy = (repeat_predictions == test_labels).float().mean().item()
    frequent_accuracy = (frequent_predictions == test_labels).float().mean().item()

    print("\nBest validation accuracy: {:.2%} (epoch {})".format(best_accuracy, best_epoch))
    print("Test accuracy: {:.2%}".format(test_accuracy))
    print("Repeat-last baseline: {:.2%}".format(repeat_accuracy))
    print("Most-frequent baseline: {:.2%}".format(frequent_accuracy))
    print("\nPer-pattern test accuracy:")
    per_pattern = {}
    for pattern in PATTERN_TYPES:
        mask = test_tags == pattern
        accuracy = (test_predictions[mask] == test_labels[mask]).float().mean().item()
        repeat = (repeat_predictions[mask] == test_labels[mask]).float().mean().item()
        frequent = (frequent_predictions[mask] == test_labels[mask]).float().mean().item()
        per_pattern[pattern] = (accuracy, repeat, frequent)
        print(
            "{}: model {:.2%} | repeat-last {:.2%} | most-frequent {:.2%} (n={})".format(
                pattern, accuracy, repeat, frequent, int(mask.sum())
            )
        )

    epochs = range(1, len(train_losses) + 1)
    figure, loss_axis = plt.subplots(figsize=(7, 4))
    loss_axis.plot(epochs, train_losses, marker="o", label="Train loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Cross-entropy loss")
    accuracy_axis = loss_axis.twinx()
    accuracy_axis.plot(epochs, val_accuracies, color="tab:orange", marker="s", label="Val accuracy")
    accuracy_axis.set_ylabel("Validation accuracy")
    loss_axis.set_title("Training curves")
    loss_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig("curves.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(PATTERN_TYPES))
    width = 0.25
    for offset, (name, column, color) in enumerate(
        (
            ("Model", 0, "tab:blue"),
            ("Repeat last", 1, "tab:orange"),
            ("Most frequent", 2, "tab:green"),
        )
    ):
        values = [per_pattern[pattern][column] for pattern in PATTERN_TYPES]
        axis.bar(positions + (offset - 1) * width, values, width, label=name, color=color)
    axis.set_xticks(positions, PATTERN_TYPES, rotation=25, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test accuracy")
    axis.set_title("Accuracy by test pattern")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig("per_pattern.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
