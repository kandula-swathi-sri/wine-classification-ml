from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_wine


MODEL_PATH = Path(
    "models/best_wine_classifier.joblib"
)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python train.py first."
        )

    model = joblib.load(MODEL_PATH)

    wine = load_wine(as_frame=True)

    sample = wine.data.iloc[[0]]

    prediction = int(
        model.predict(sample)[0]
    )

    probabilities = model.predict_proba(
        sample
    )[0]

    result = {
        "predicted_class": wine.target_names[
            prediction
        ],
        "probabilities": {
            class_name: round(
                float(probability),
                4,
            )
            for class_name, probability in zip(
                wine.target_names,
                probabilities,
            )
        },
    }

    print("Sample Input:")
    print(pd.DataFrame(sample))

    print("\nPrediction:")
    print(result)


if __name__ == "__main__":
    main()
