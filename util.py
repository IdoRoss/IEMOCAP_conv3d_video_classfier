import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def get_lable(path):
    return path.split(".")[0].split("-")[-1]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_model(model, batches, test_type, classes):
    sum_correct = 0
    print("Indices:", batches.class_indices)

    print(f"{test_type}set labels:", batches.classes)

    predictions = model.predict(x=batches, verbose=0)
    print("Predictions:", end=" ")
    predictions = np.round(predictions)
    for i in range(len(predictions)):
        pred = predictions[i]
        idx = np.argmax(pred, axis=-1)
        print(idx, end=" ")
        if idx == batches.classes[i]:
            sum_correct += 1
    print("\n")
    acc = float(sum_correct / len(predictions))
    acc = format(acc, ".3f")
    print("Accuracy:", acc)
    y_pred = np.argmax(predictions, axis=-1)
    cm = confusion_matrix(y_true=batches.classes, y_pred=y_pred)

    plot_confusion_matrix(cm=cm, classes=classes, title=f"{test_type} Set, {acc}% Accuracy")
    print("\n")