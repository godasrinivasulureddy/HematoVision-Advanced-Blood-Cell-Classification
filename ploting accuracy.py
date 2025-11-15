import matplotlib.pyplot as plt
import json


def plot_training(history_path="history.json"):
    with open(history_path, "r") as f:
        hist = json.load(f)

    acc = hist["accuracy"]
    val_acc = hist["val_accuracy"]
    loss = hist["loss"]
    val_loss = hist["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc, label="Training Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_training()
