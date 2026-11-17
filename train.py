import os
import glob
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from s_generator import generate_full_dataset
from model import build_chirality_cnn
from data_utils import (
    BASE_DIR,
    S_DIR,
    ANTI_S_DIR,
    chirality_pseudo_labels,
    get_train_test_paths,
    load_images,
)


def save_sample_grid(s_dir, anti_s_dir, save_path="training_samples.png"):
    """Outputs a 10-image visual check (5 S, 5 Anti-S)."""
    s_files = sorted(glob.glob(os.path.join(s_dir, "*.png")))[:5]
    anti_files = sorted(glob.glob(os.path.join(anti_s_dir, "*.png")))[:5]

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
    fig.suptitle("Generated Training Samples (Top: S | Bottom: Anti-S)", fontsize=12)

    for i in range(5):
        axes[0, i].imshow(Image.open(s_files[i]))
        axes[0, i].set_title(f"S #{i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(Image.open(anti_files[i]))
        axes[1, i].set_title(f"Anti-S #{i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Saved 10 sample images to '{save_path}'.")


def train():
    s_dir = str(S_DIR)
    anti_s_dir = str(ANTI_S_DIR)

    # 1. Generate paired images if either side of the dataset is incomplete.
    if (
        len(glob.glob(os.path.join(s_dir, "*.png"))) < 5000
        or len(glob.glob(os.path.join(anti_s_dir, "*.png"))) < 5000
    ):
        generate_full_dataset(num_samples=5000)
    else:
        print("✓ Found existing paired image dataset.")

    # 2. Save 10 sample images
    save_sample_grid(s_dir, anti_s_dir, save_path=str(BASE_DIR / "training_samples.png"))

    # 3. Keep test pairs out of training, while not using their class labels.
    train_s, train_anti_s, _, _ = get_train_test_paths()
    all_paths = train_s + train_anti_s
    random.Random(42).shuffle(all_paths)
    x_all = load_images(all_paths)
    print(f"✓ Loaded {len(x_all)} unlabeled images into RAM ({x_all.nbytes / 1e6:.1f} MB).")

    # 4. Learn pseudo-labels from the rotation-invariant signed shape moment.
    pseudo_labels = chirality_pseudo_labels(x_all)

    # 5. Train a compact classifier on the unlabeled images and geometric targets.
    network = build_chirality_cnn(input_shape=(35, 35, 1))
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=1e-4,
        patience=2,
        restore_best_weights=True,
    )

    network.fit(
        x_all,
        pseudo_labels,
        batch_size=256,
        epochs=20,
        shuffle=True,
        callbacks=[early_stop],
        verbose=1
    )

    network.save(BASE_DIR / "chirality_cnn_tf.keras")
    print(f"✓ Saved trained model to '{BASE_DIR / 'chirality_cnn_tf.keras'}'.")


if __name__ == "__main__":
    train()