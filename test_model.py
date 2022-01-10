import tensorflow as tf
from tensorflow import keras
import numpy as np
from util import *

def test_model(model, batches,classes):
    batch_size_tmp = batches.batch_size
    batches.batch_size=(len(batches)+1)*batches.batch_size
    X,y = batches[0]
    predictions = model.predict(x=X)
    y_pred = np.argmax(predictions, axis=-1)
    y_real = np.argmax(y, axis=-1)
    cm = confusion_matrix(y_true=y_real, y_pred=y_pred)
    sum_correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_real[i]:
            sum_correct+=1

    acc = float(sum_correct / len(y_pred))
    acc = format(acc, ".3f")
    plot_confusion_matrix(cm=cm, classes=classes, title=f"{acc}% Accuracy")
    batches.batch_size =batch_size_tmp