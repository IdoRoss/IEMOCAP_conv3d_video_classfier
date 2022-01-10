from data_prep import *
from c3dmodel import *
from test_model import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from keras.models import load_model
from keras_video import VideoFrameGenerator

def train_c3d_model(model, model_name, epochs, train, valid):

    NUM_EPOCHS = epochs
    optimizer = keras.optimizers.Adam(0.001)
    loss = "categorical_crossentropy"
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]

    print("---------------------------------------Training Model---------------------------------------")
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.fit(train, validation_data=valid, epochs=NUM_EPOCHS,
                        callbacks=callbacks, verbose=1, shuffle=True)
    print("---------------------------------------Saving Model---------------------------------------")
    model.save(f"models/{model_name}.h5")