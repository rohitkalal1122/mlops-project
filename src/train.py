import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from src.preprocess import load_and_preprocess_data
except ModuleNotFoundError:
    from preprocess import load_and_preprocess_data


def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    joblib.dump(model, "model.pkl")

    print("Model saved as model.pkl")


if __name__ == "__main__":
    train_model()
