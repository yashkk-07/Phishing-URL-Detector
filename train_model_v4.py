
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from train_model_v3 import extract_host_features

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "additional_urls_v4.csv"
MODEL_PATH = BASE_DIR / "phishing_model_v4.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[["URL", "Label"]].dropna().copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["Label"])
    df = df[df["Label"].isin([0, 1])]
    df["Label"] = df["Label"].astype(int)
    df = df[df["URL"] != ""].drop_duplicates("URL")

    df["Host"] = df["URL"].map(
        lambda u: __import__("urllib.parse", fromlist=["urlparse"])
        .urlparse(u).hostname or ""
    )
    df = df[df["Host"] != ""].reset_index(drop=True)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    train_idx, test_idx = next(
        splitter.split(df, df["Label"], groups=df["Host"])
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    feature_cols = [
        "HostLength", "HostIsIP", "SubdomainCount",
        "HasHyphenInHost", "HostDigitCount",
        "PathLength", "QueryLength", "URLLength",
    ]

    train_features = pd.DataFrame(
        train_df["URL"].map(extract_host_features).tolist(),
        index=train_df.index,
    )
    test_features = pd.DataFrame(
        test_df["URL"].map(extract_host_features).tolist(),
        index=test_df.index,
    )

    X_train = pd.concat(
        [train_df[["URL"]], train_features[feature_cols]], axis=1
    )
    X_test = pd.concat(
        [test_df[["URL"]], test_features[feature_cols]], axis=1
    )

    y_train = train_df["Label"]
    y_test = test_df["Label"]

    preprocessing = ColumnTransformer([
        (
            "url_text",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 6),
                min_df=3,
                max_features=150000,
                sublinear_tf=True,
            ),
            "URL",
        ),
        (
            "url_features",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            feature_cols,
        ),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("Training V4...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0,
        digits=4,
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    print(f"\nAccuracy: {accuracy:.4%}")
    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix:")
    print(cm)

    # Create the reports folder
    REPORTS_DIR = BASE_DIR / "reports"
    REPORTS_DIR.mkdir(exist_ok=True)

    # Save the classification report
    with open(
        REPORTS_DIR / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Accuracy: {accuracy:.4%}\n\n")
        file.write("Classification Report\n")
        file.write(report)
        file.write("\nConfusion Matrix\n")
        file.write(str(cm))
        file.write("\n\nEvaluation setup: GroupShuffleSplit, "
                   "test_size=0.20, random_state=42\n")

    # Plot and save the confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        title="Phishing URL Detector - Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="Actual Label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Phishing", "Legitimate"],
        yticklabels=["Phishing", "Legitimate"],
    )

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col,
                row,
                f"{cm[row, col]:,}",
                ha="center",
                va="center",
            )

    fig.tight_layout()
    fig.savefig(
        REPORTS_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"\nSaved evaluation files to: {REPORTS_DIR}")


    joblib.dump(model, MODEL_PATH)
    print("\nSaved new model:", MODEL_PATH)

    print("\nKnown URL diagnostics:")
    test_urls = [
        "https://www.google.com/",
        "https://drive.google.com/",
        "https://docs.google.com/",
        "https://www.youtube.com/",
        "https://www.amazon.in/",
    ]

    for url in test_urls:
        features = extract_host_features(url)
        sample = pd.DataFrame([{"URL": url, **features}])
        probabilities = model.predict_proba(sample)[0]
        classes = list(model.classes_)
        phishing_prob = probabilities[classes.index(0)]
        prediction = model.predict(sample)[0]

        print(
            f"{url} -> "
            f"{'Phishing' if prediction == 0 else 'Legitimate'}; "
            f"phishing probability: {phishing_prob:.2%}"
        )


if __name__ == "__main__":
    main()
