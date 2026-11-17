Unlabeled Image Classifier (`S` vs. mirrored `Anti-S`)

A TensorFlow/Keras Convolutional Neural Network trained to binary classify standard `'S'` characters (`1`) from horizontally mirrored `'Anti-S'` characters (`0`) across arbitrary $360^\circ$ rotations without using ground truth class labels during training.

---

Methodology & Architecture

Purely unsupervised training based solely on rotation invariance and mirror differences fails due to angle partitioning degeneracy where labels are assigned based on arbitrary learned geometric trends in the images (e.g. leaning right or left) and simplicity bias on continuous $360^\circ$ rotations. To combat these issues, this model combines geometric pseudo-labeling with a Siamese anti-symmetric CNN architecture:

1. Geometric Pseudo-Labeling (`data_utils.py`):
    *Unlabeled training images are assigned binary targets using a rotation-invariant signed shape moment (`chirality_pseudo_labels`), making a strong guess of the global handedness of the curve without ground-truth labels.
2. Siamese Anti-Symmetric Network (`model.py`):**
   *`HorizontalMirror`:** Duplicates every input image $x$ with its horizontal reflection $\text{flip}(x)$ inside the forward pass.
   * Shared Convolutional Backbone:** Extracts scalar scores $s(x)$ and $s(\text{flip}(x))$ using shared weights.
   *`ReflectionDifference`:** Computes the logit $z(x) = s(x) - s(\text{flip}(x))$ before applying a `sigmoid` activation, for complementary probabilities $(p(\text{flip}(x)) = 1 - p(x))$ by construction.
3. Training (`train.py`):
   * Optimized with Binary Cross-Entropy (`binary_crossentropy`) and `Adam` (`lr=3e-3`), pushing mirrored pairs toward opposite ends of the $[0, 1]$ probability spectrum.

---

## Project Structure

```text
S_classifier_unlabeled/
├── s_generator.py            # Generates 35x35 rotated 'S' and mirrored 'Anti-S' PNGs
├── data_utils.py             # Train/test splitting, image loading, and geometric pseudo-labels
├── model.py                  # Siamese ChiralCNN with HorizontalMirror & ReflectionDifference
├── train.py                  # Unlabeled training pipeline & sample grid generation
├── test.py                   # Held-out labeled evaluation (confusion matrix & histogram)
├── requirements.txt          # Python dependencies
├── training_samples.png      # Sample generated 'S' and 'Anti-S' inputs
├── confusion_matrix.png      # Test set classification results (n = 1,000)
└── prediction_histogram.png  # Distribution of predicted probabilities [0, 1]
