import joblib


def predict(input_data):
    model = joblib.load("model.pkl")

    prediction = model.predict([input_data])

    return prediction[0]


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict(sample)

    print(f"Predicted class: {result}")
