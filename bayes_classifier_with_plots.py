#!/usr/bin/env python3
"""
bayes_classifier_with_plots.py

Bayesian Gaussian classifier (LDA/QDA) + visualizations:
 - Confusion matrix heatmap
 - Decision boundary plot for 2D data (or PCA projection to 2D)

Usage examples:
  python bayes_classifier_with_plots.py --model qda
  python bayes_classifier_with_plots.py --data my.csv --target label --plot-pca --save-plots output_prefix
"""
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns

EPS = 1e-6  # regularization for covariance to avoid singular matrices


def load_data(csv_path=None, target=None):
    if csv_path is None:
        from sklearn.datasets import load_iris
        iris = load_iris()
        X = iris["data"]
        y = iris["target"]
        feature_names = iris["feature_names"]
        return X, y, feature_names
    else:
        df = pd.read_csv(csv_path)
        if target is None:
            raise ValueError("When using --data, you must pass --target with the target column name.")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in CSV columns: {df.columns.tolist()}")
        X = df.drop(columns=[target]).values
        y = df[target].values
        feature_names = [c for c in df.columns if c != target]
        return X, y, feature_names


class BayesGaussianClassifier:
    def __init__(self, shared_cov=True, reg=EPS):
        self.shared_cov = bool(shared_cov)
        self.reg = float(reg)
        self.classes_ = None
        self.priors_ = {}
        self.means_ = {}
        self.covariances_ = {}
        self.shared_covariance_ = None
        self.fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n, d = X.shape
        self.classes_ = classes

        for c in classes:
            Xc = X[y == c]
            self.priors_[c] = Xc.shape[0] / n
            self.means_[c] = Xc.mean(axis=0)
            cov = np.cov(Xc, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[cov + self.reg]])
            cov += np.eye(cov.shape[0]) * self.reg
            self.covariances_[c] = cov

        if self.shared_cov:
            pooled = np.zeros_like(self.covariances_[classes[0]])
            for c in classes:
                nc = np.sum(y == c)
                pooled += (nc - 1) * (self.covariances_[c] - np.eye(self.covariances_[c].shape[0]) * self.reg)
            pooled /= (n - len(classes))
            pooled += np.eye(pooled.shape[0]) * self.reg
            self.shared_covariance_ = pooled

        self.fitted = True
        return self

    def _log_gaussian(self, x, mean, cov):
        d = mean.shape[0]
        xm = x - mean
        try:
            sol = np.linalg.solve(cov, xm.T)
            quad = np.sum(xm * sol.T, axis=-1)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                cov = cov + np.eye(d) * (self.reg * 10)
                sign, logdet = np.linalg.slogdet(cov)
            log_norm = -0.5 * (d * np.log(2 * np.pi) + logdet)
            return log_norm - 0.5 * quad
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
            quad = np.sum(xm @ inv * xm, axis=-1)
            sign, logdet = np.linalg.slogdet(cov + np.eye(d) * self.reg)
            return -0.5 * (d * np.log(2 * np.pi) + logdet) - 0.5 * quad

    def predict_log_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit(X, y) first.")
        X = np.asarray(X)
        n = X.shape[0]
        K = len(self.classes_)
        log_scores = np.zeros((n, K))
        for idx, c in enumerate(self.classes_):
            mu = self.means_[c]
            cov = self.shared_covariance_ if self.shared_cov else self.covariances_[c]
            log_likelihood = self._log_gaussian(X, mu, cov)
            log_prior = np.log(self.priors_.get(c, 1e-12))
            log_scores[:, idx] = log_likelihood + log_prior
        return log_scores

    def predict(self, X):
        log_scores = self.predict_log_proba(X)
        indices = np.argmax(log_scores, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        log_scores = self.predict_log_proba(X)
        a_max = np.max(log_scores, axis=1, keepdims=True)
        exp = np.exp(log_scores - a_max)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    report = metrics.classification_report(y_test, preds, zero_division=0)
    cm = metrics.confusion_matrix(y_test, preds)
    return acc, report, cm, preds


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", savepath=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


def plot_decision_boundary_2d(clf, X, y, feature_names=None, title="Decision Boundary", savepath=None, resolution=0.02):
    """
    Expects X to be (n_samples, 2). clf must implement predict(X).
    """
    X = np.asarray(X)
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    try:
        Z = clf.predict(grid)
    except Exception as e:
        # If the classifier expects original feature dims, raise
        raise RuntimeError("Classifier predict failed on 2D grid: " + str(e))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    # TODO: use default colormap; seaborn will style the plots
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor="k")
    if feature_names and len(feature_names) >= 2:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    plt.title(title)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Bayesian Gaussian classifier (LDA/QDA) with plots")
    parser.add_argument("--data", type=str, default=None, help="CSV file path (if omitted, loads Iris dataset)")
    parser.add_argument("--target", type=str, default=None, help="Target column name in CSV")
    parser.add_argument("--model", type=str, default="qda", choices=["qda", "lda"], help="qda=separate cov (QDA), lda=shared cov (LDA)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-model", type=str, default=None, help="Optional path to save trained model (pickle)")
    parser.add_argument("--plot-pca", action="store_true", help="If data is not 2D, project to 2D using PCA for plotting")
    parser.add_argument("--save-plots", type=str, default=None, help="Prefix for saving plots (e.g. 'out/myplot' -> out/myplot_confmat.png)")
    args = parser.parse_args()

    try:
        X, y, feature_names = load_data(args.data, args.target)
    except Exception as e:
        print("Error loading data:", e)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    shared_cov = args.model.lower() == "lda"
    clf = BayesGaussianClassifier(shared_cov=shared_cov)
    clf.fit(X_train, y_train)

    acc, report, cm, preds = evaluate_model(clf, X_test, y_test)

    print("=" * 60)
    print(f"Model: {'LDA (shared covariance)' if shared_cov else 'QDA (class-specific covariance)'}")
    print(f"Dataset: {'Iris (builtin)' if args.data is None else args.data}")
    print(f"Features: {feature_names}")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 60)
    print("Classification report:")
    print(report)
    print("Confusion matrix:")
    print(cm)
    print("=" * 60)

    if args.save_model:
        with open(args.save_model, "wb") as f:
            pickle.dump(clf, f)
        print(f"Saved model to {args.save_model}")

    # ----------------------------
    # Visualization
    # ----------------------------
    class_labels = np.unique(y_test)
    save_prefix = args.save_plots

    # Confusion matrix heatmap
    cm_path = None
    if save_prefix:
        cm_path = f"{save_prefix}_confusion_matrix.png"
    plot_confusion_matrix(cm, classes=class_labels, title="Confusion Matrix", savepath=cm_path)

    # Decision boundary:
    # If original X_test is 2D we can plot directly. Otherwise we can optionally PCA to 2D.
    if X.shape[1] == 2:
        db_path = None
        if save_prefix:
            db_path = f"{save_prefix}_decision_boundary.png"
        # Use the classifier trained on original features. But some classifiers need training on 2D for plotting grid predictions.
        # We'll retrain a fresh model on TRAINING data restricted to these two dims to make sure the classifier.predict works on 2D grid.
        clf_2d = BayesGaussianClassifier(shared_cov=shared_cov)
        clf_2d.fit(X_train[:, :2], y_train)
        plot_decision_boundary_2d(clf_2d, X_test[:, :2], y_test, feature_names=(feature_names[0], feature_names[1]) if feature_names else None,
                                 title="Decision Boundary (2D features)", savepath=db_path)
    else:
        if args.plot_pca:
            # Project train & test to 2D PCA and retrain on projected 2D for visualization
            pca = PCA(n_components=2, random_state=args.random_state)
            X_train_p = pca.fit_transform(X_train)
            X_test_p = pca.transform(X_test)
            clf_pca = BayesGaussianClassifier(shared_cov=shared_cov)
            clf_pca.fit(X_train_p, y_train)
            db_path = None
            if save_prefix:
                db_path = f"{save_prefix}_decision_boundary_pca.png"
            plot_decision_boundary_2d(clf_pca, X_test_p, y_test, feature_names=("PC1", "PC2"),
                                     title="Decision Boundary (PCA -> 2D)", savepath=db_path)
        else:
            print("Note: Decision boundary plot requires 2D data. Use --plot-pca to visualize via PCA 2D projection.")

if __name__ == "__main__":
    main()
