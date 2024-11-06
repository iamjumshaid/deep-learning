"""Logistic regression."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from typing import Tuple


def logistic_regression(inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, float]:
    """Implement a logistic regression model for binary classification.

    Args:
        inputs: np.ndarray with shape (nr_examples, nr_features). Input examples.
        labels: np.ndarray with shape (nr_examples). True labels.

    Returns:
    Tuple[prediction, score]:
        prediction: np.ndarray with shape (nr_examples). Predicted labels.
        score: float. Accuracy of the model on the input data.
    """
    # START TODO #################
    # Create model
    model = LogisticRegression(solver='lbfgs', penalty='l2')

    # Fit the model to the training data
    model.fit(inputs, labels)

    # Calculate accuracy on the test set
    prediction = model.predict(inputs)
    score = accuracy_score(labels, prediction)

    # END TODO ##################
    print(f"Prediction: {prediction}")
    print(f"Score: {score}")
    return prediction, score
