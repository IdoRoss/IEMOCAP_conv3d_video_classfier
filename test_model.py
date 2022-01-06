import tensorflow as tf
from tensorflow import keras
import numpy as np
from util import *

def test_model(model, batches,classes):
    predictions = model.predict(x=batches)
    y_pred = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(y_true=batches.classes, y_pred=y_pred)
    sum_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == batches.classes[i]:
            sum_correct+=1

    acc = float(sum_correct / len(predictions))
    acc = format(acc, ".3f")
    plot_confusion_matrix(cm=cm, classes=classes, title=f"{acc}% Accuracy")