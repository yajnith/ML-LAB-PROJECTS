#!/usr/bin/env python3
"""
mnist_binary_classifier.py

Train a binary classifier on MNIST (choose two digits).

Examples:
  # default (0 vs 1)
  python mnist_binary_classifier.py

  # train 3 vs 8 with PCA to 50 dims and save model
  python mnist_binary_classifier.py --pos 3 --neg 8 --pca 50 --save-model model_joblib.pkl
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
import sys

def load_mnist_two_classes(pos_digit=0, neg_digit=1, subsample=None, random_state=42):
    """Fetch MNIST and return X, y filtered to two digits.
    y will be binary: 1 for pos_digit, 0 for neg_digit.
    subsample: if int, randomly sample that many examples total (for speed)."""
    print("📥 Downloading/fetching MNIST (may take a minute if not cached)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist["data"]     # shape (70000, 784)
    y = mnist["target"].astype(int)  # shape (70000,)
    # Filter to two digits
    mask = (y == pos_digit) | (y == neg_digit)
    X = X[mask]
    y = y[mask]
    y_bin = (y == pos_digit).astype(int)
    # optional subsample for speed
    if subsample is not None and subsample < X.shape[0]:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(np.arange(X.shape[0]), size=subsample, replace=False)
        X = X[idx]
        y_bin = y_bin[idx]
    print(f"Loaded MNIST subset: {X.shape[0]} samples of digits {pos_digit} and {neg_digit}")
    return X, y_bin

def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=None, savepath=None):
    plt.figure(figsize=(5,4))
    if cmap is None:
        cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(int(cm[i, j]), fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()

def show_sample_predictions(X_orig, X_transformed, y_true, y_pred, pos_digit, neg_digit, n_show=8):
    """Show a few example test digits with predicted vs true."""
    # pick n_show random indices
    rng = np.random.RandomState(0)
    idx = rng.choice(np.arange(len(y_true)), size=min(n_show, len(y_true)), replace=False)
    plt.figure(figsize=(12, 3))
    for i, j in enumerate(idx):
        ax = plt.subplot(1, len(idx), i+1)
        img = X_orig[j].reshape(28,28)
        ax.imshow(img, cmap='gray_r')
        true_label = pos_digit if y_true[j]==1 else neg_digit
        pred_label = pos_digit if y_pred[j]==1 else neg_digit
        ax.set_title(f"T:{true_label}\nP:{pred_label}", color=("green" if y_true[j]==y_pred[j] else "red"))
        ax.axis('off')
    plt.suptitle("Sample predictions (green=correct, red=wrong)")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Binary MNIST classifier (Logistic Regression)")
    parser.add_argument("--pos", type=int, default=0, help="Positive digit (default 0)")
    parser.add_argument("--neg", type=int, default=1, help="Negative digit (default 1)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--pca", type=int, default=0, help="If >0 apply PCA to this many components (speed/visual)")
    parser.add_argument("--subsample", type=int, default=None, help="Subsample total training+test size for speed (e.g., 5000)")
    parser.add_argument("--save-model", type=str, default=None, help="If set, save trained model with joblib")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for logistic regression")
    args = parser.parse_args()

    if args.pos == args.neg:
        print("Error: --pos and --neg must be different digits (0-9).")
        sys.exit(1)
    if not (0 <= args.pos <= 9 and 0 <= args.neg <= 9):
        print("Error: digits must be between 0 and 9.")
        sys.exit(1)

    # 1) Load data
    X, y = load_mnist_two_classes(pos_digit=args.pos, neg_digit=args.neg, subsample=args.subsample, random_state=args.random_state)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # 3) Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(float))
    X_test_scaled = scaler.transform(X_test.astype(float))

    # 4) Optional PCA
    pca = None
    if args.pca and args.pca > 0:
        print(f"🔽 Applying PCA -> {args.pca} components for speed/visualization")
        pca = PCA(n_components=args.pca, random_state=args.random_state)
        X_train_proc = pca.fit_transform(X_train_scaled)
        X_test_proc = pca.transform(X_test_scaled)
    else:
        X_train_proc = X_train_scaled
        X_test_proc = X_test_scaled

    print("Training Logistic Regression (binary)...")
    clf = LogisticRegression(solver='lbfgs', max_iter=args.max_iter, random_state=args.random_state)
    clf.fit(X_train_proc, y_train)

    # 5) Evaluate
    y_pred = clf.predict(X_test_proc)
    acc = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred, target_names=[str(args.neg), str(args.pos)], zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    print("\n=== Results ===")
    print(f"Digits: pos={args.pos} (label=1), neg={args.neg} (label=0)")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)

    # 6) Plots
    plot_confusion_matrix(cm, classes=[str(args.neg), str(args.pos)], title="Confusion matrix (neg, pos)")

    # show some sample test images and predictions
    show_sample_predictions(X_test, X_test_proc, y_test, y_pred, args.pos, args.neg, n_show=8)

    # 7) Save model & preprocessing (if requested)
    if args.save_model:
        payload = {
            "model": clf,
            "scaler": scaler,
            "pca": pca,
            "pos_digit": args.pos,
            "neg_digit": args.neg
        }
        joblib.dump(payload, args.save_model)
        print(f"Saved model+preprocessing to {args.save_model}")

    print("Done.")

if __name__ == "__main__":
    main()
