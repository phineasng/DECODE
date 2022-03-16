import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def batch_predict(dataset, model, batch_size=None):

    if batch_size is None:
        batch_size = len(dataset)

    progress = range(0, len(dataset), batch_size)

    predictions = []
    for i in progress:
        x = dataset[i:i+batch_size]
        predictions.append(model.predict(x))

    predictions = np.concatenate(predictions, axis=0)
    return predictions


def test_model(dataset, labels, model, batch_size):
    predictions = batch_predict(dataset, model, batch_size)
    return accuracy_score(labels, predictions), confusion_matrix(labels, predictions)
