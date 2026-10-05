import os

from src.preprocess import load_and_preprocess_data
from src.train import train_model


def test_data_loading():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_model_training():
    train_model()

    assert os.path.exists("model.pkl")
