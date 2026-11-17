import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from model import HorizontalMirror, ReflectionDifference
from data_utils import BASE_DIR, get_train_test_paths, load_images


def evaluate():
    # 1. Evaluate on held-out image pairs never used for training.
    _, _, s_files, anti_s_files = get_train_test_paths()
    test_paths = anti_s_files + s_files
    y_true = np.array([0] * len(anti_s_files) + [1] * len(s_files))

    # 2. Load the trained model
    model = tf.keras.models.load_model(BASE_DIR / "chirality_cnn_tf.keras")

    # 3. Run vectorized inference
    x_test = load_images(test_paths)
    probs = model.predict(x_test, batch_size=256, verbose=0).squeeze(-1)

    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy (1,000 labeled samples): {acc * 100:.2f}%")

    # 4. Confusion Matrix (0 = Anti-S, 1 = S)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Anti-S (0)", "S (1)"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix (Accuracy: {acc * 100:.2f}%)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print("✓ Saved 'confusion_matrix.png'.")

    # 5. Binned Histogram of Predicted Probabilities [0.0 to 1.0]
    anti_s_probs = probs[y_true == 0]
    s_probs = probs[y_true == 1]

    bins = np.linspace(0.0, 1.0, 41)  # 40 bins of width 0.025

    plt.figure(figsize=(9, 5))
    plt.hist(anti_s_probs, bins=bins, alpha=0.75, color="#d95f02", edgecolor="black", label="True Anti-S (Target: 0)")
    plt.hist(s_probs, bins=bins, alpha=0.75, color="#1b9e77", edgecolor="black", label="True S (Target: 1)")

    plt.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Decision Boundary (0.5)")
    plt.xlim(-0.02, 1.02)
    plt.xlabel("Predicted Network Output Probability [0 to 1]", fontsize=11)
    plt.ylabel("Image Count", fontsize=11)
    plt.title("Distribution of Predicted Chirality Probabilities (Test Set: n=1,000)", fontsize=12)
    plt.legend(loc="upper center")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(BASE_DIR / "prediction_histogram.png", dpi=150)
    plt.close()
    print("✓ Saved 'prediction_histogram.png'.")


if __name__ == "__main__":
    evaluate()