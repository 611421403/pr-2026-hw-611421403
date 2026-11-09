from pathlib import Path
import csv
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.exceptions import ConvergenceWarning

from gaussian_nb import GaussianNBScratch


SEED = 0
TEST_SIZE = 0.30
QDA_REG_PARAMS = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
# Two fixed pixel coordinates (row, column) in the 8x8 image.
PIXEL_COORDS = [(2, 3), (5, 4)]
PIXEL_INDICES = [r * 8 + c for r, c in PIXEL_COORDS]

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180)
    plt.close(fig)


def plot_qda_regularization(reg_values, accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(reg_values, accuracies, marker="o")
    ax.set_xscale("symlog", linthresh=1e-6)
    ax.set_xlabel("QDA reg_param")
    ax.set_ylabel("Test accuracy")
    ax.set_title("QDA test accuracy vs. regularization strength")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qda_regularization.png", dpi=180)
    plt.close(fig)


def plot_decision_regions(X_train_2d, y_train, models):
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    class_labels = np.unique(y_train)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        pred = model.predict(grid).reshape(xx.shape)
        ax.contourf(
            xx, yy, pred,
            levels=np.arange(class_labels.min() - 0.5, class_labels.max() + 1.5, 1),
            cmap="tab10", alpha=0.22,
        )
        scatter = ax.scatter(
            X_train_2d[:, 0], X_train_2d[:, 1],
            c=y_train, cmap="tab10", s=10, edgecolors="none",
        )
        ax.set_title(f"{name} decision regions")
        ax.set_xlabel(f"Pixel {PIXEL_COORDS[0]} (feature {PIXEL_INDICES[0]})")
        ax.set_ylabel(f"Pixel {PIXEL_COORDS[1]} (feature {PIXEL_INDICES[1]})")
        ax.grid(True, alpha=0.15)
    handles, _ = scatter.legend_elements()
    fig.legend(handles, [str(c) for c in class_labels], title="Digit class",
               loc="lower center", ncol=10)
    fig.suptitle("Decision regions using two pixel features", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    fig.savefig(FIG_DIR / "decision_regions_2d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    np.random.seed(SEED)
    digits = load_digits()
    X = digits.data.astype(np.float64)
    y = digits.target.astype(int)
    labels = np.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Train full 64-dimensional models.
    models = {
        "GaussianNB (scratch)": GaussianNBScratch(var_smoothing=1e-9),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA (reg=0.01)": QuadraticDiscriminantAnalysis(reg_param=0.01),
    }

    rows = []
    predictions = {}
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        predictions[name] = pred
        acc = accuracy_score(y_test, pred)
        cm = confusion_matrix(y_test, pred, labels=labels)
        rows.append({"model": name, "test_accuracy": float(acc)})
        safe_name = (
            name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("=", "").replace(".", "")
        )
        save_confusion_matrix(
            cm, labels, f"{name} confusion matrix",
            f"confusion_matrix_{safe_name}.png",
        )

    # Compare our implementation directly with sklearn GaussianNB.
    sklearn_nb = GaussianNB(var_smoothing=1e-9)
    sklearn_nb.fit(X_train, y_train)
    sklearn_pred = sklearn_nb.predict(X_test)
    scratch_pred = predictions["GaussianNB (scratch)"]
    comparison = {
        "scratch_accuracy": float(accuracy_score(y_test, scratch_pred)),
        "sklearn_accuracy": float(accuracy_score(y_test, sklearn_pred)),
        "prediction_agreement": float(np.mean(scratch_pred == sklearn_pred)),
        "max_abs_log_proba_difference": float(np.max(np.abs(
            models["GaussianNB (scratch)"].predict_log_proba(X_test)
            - sklearn_nb.predict_log_proba(X_test)
        ))),
    }

    # QDA regularization sweep. Each candidate is fitted on training data only.
    reg_accuracies = []
    for reg in QDA_REG_PARAMS:
        qda = QuadraticDiscriminantAnalysis(reg_param=float(reg))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                qda.fit(X_train, y_train)
                pred = qda.predict(X_test)
            reg_accuracies.append(float(accuracy_score(y_test, pred)))
        except np.linalg.LinAlgError:
            # With reg_param=0, a class covariance can be rank-deficient.
            # Keep the failed setting in the sweep as NaN rather than aborting.
            reg_accuracies.append(float("nan"))
    plot_qda_regularization(QDA_REG_PARAMS, reg_accuracies)

    # Two-dimensional experiment: refit all models on the same two features.
    X_train_2d = X_train[:, PIXEL_INDICES]
    X_test_2d = X_test[:, PIXEL_INDICES]
    models_2d = {
        "GaussianNB": GaussianNBScratch(var_smoothing=1e-9),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),
    }
    for model in models_2d.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_2d, y_train)
    plot_decision_regions(X_train_2d, y_train, models_2d)

    # Store machine-readable results for the report.
    with (RESULT_DIR / "accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "test_accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    with (RESULT_DIR / "qda_regularization.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["reg_param", "test_accuracy"])
        writer.writerows(zip(QDA_REG_PARAMS, reg_accuracies))

    summary = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "class_count": int(len(labels)),
        "accuracy": rows,
        "qda_regularization": [
            {"reg_param": float(r), "test_accuracy": (float(a) if np.isfinite(a) else None)}
            for r, a in zip(QDA_REG_PARAMS, reg_accuracies)
        ],
        "best_qda_reg_param": float(QDA_REG_PARAMS[int(np.nanargmax(reg_accuracies))]),
        "best_qda_accuracy": float(np.nanmax(reg_accuracies)),
        "gaussian_nb_comparison": comparison,
        "two_pixel_features": PIXEL_INDICES,
        "two_pixel_coordinates": PIXEL_COORDS,
    }
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== Test accuracy ===")
    for row in rows:
        print(f"{row['model']:<24}: {row['test_accuracy']:.4f}")

    print("\n=== GaussianNB implementation comparison ===")
    for key, value in comparison.items():
        print(f"{key}: {value:.8f}")

    print("\n=== QDA regularization sweep ===")
    for reg, acc in zip(QDA_REG_PARAMS, reg_accuracies):
        print(f"reg_param={reg:<8g} accuracy={acc:.4f}")

    best_idx = int(np.nanargmax(reg_accuracies))
    print("\nBest QDA setting:",
          QDA_REG_PARAMS[best_idx],
          "accuracy:", reg_accuracies[best_idx])
    if np.isnan(reg_accuracies[0]):
        print("Note: QDA reg_param=0 failed because at least one class covariance was rank-deficient.")
    print(f"\nFigures saved to: {FIG_DIR}")
    print(f"Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
