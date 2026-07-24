from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    wine = load_wine(as_frame=True)

    dataframe = wine.frame.copy()

    features = dataframe.drop(columns=["target"])
    target = dataframe["target"]

    return features, target, wine.target_names


def build_models():
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }

    return models


def main():
    features, target, target_names = load_data()

    print("Dataset shape:", features.shape)
    print("Target classes:", list(target_names))

    print("\nMissing values:")
    print(features.isnull().sum())

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    models = build_models()

    results = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        report = classification_report(
            y_test,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        results[model_name] = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(
                float(report["weighted avg"]["precision"]),
                4,
            ),
            "recall": round(
                float(report["weighted avg"]["recall"]),
                4,
            ),
            "f1_score": round(
                float(report["weighted avg"]["f1-score"]),
                4,
            ),
        }

        trained_models[model_name] = model

        print("\n", model_name)
        print("-" * 40)
        print("Accuracy:", round(accuracy, 4))

        print(
            classification_report(
                y_test,
                predictions,
                target_names=target_names,
                zero_division=0,
            )
        )

    best_model_name = max(
        results,
        key=lambda name: results[name]["accuracy"],
    )

    best_model = trained_models[best_model_name]

    best_predictions = best_model.predict(x_test)

    joblib.dump(
        best_model,
        MODEL_DIR / "best_wine_classifier.joblib",
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_model": best_model_name,
                "results": results,
            },
            file,
            indent=4,
        )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_predictions,
        display_labels=target_names,
        cmap="Blues",
    )

    plt.title(
        f"Confusion Matrix - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    random_forest = trained_models["Random Forest"]

    feature_importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    top_features = feature_importance.head(10).sort_values(
        "importance"
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance"],
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(
        "Top 10 Random Forest Feature Importances"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=300,
    )

    plt.close()

    comparison = pd.DataFrame(
        results
    ).T.sort_values(
        "accuracy",
        ascending=False,
    )

    print("\nModel Comparison:")
    print(comparison)

    print("\nBest Model:", best_model_name)

    print(
        "Saved model to models/best_wine_classifier.joblib"
    )

    print(
        "Saved results and graphs to outputs folder"
    )


if __name__ == "__main__":
    main()
