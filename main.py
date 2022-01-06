from data_prep import *
from c3dmodel import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

def main():
    # use GPU
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print("Num GPUs Available: ", len(physical_devices))
    if len(physical_devices) != 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # vars

    classes = ["ang", "fru", "hap", "sad"]
    data_dir = "D:\Code\Colman\Research\Datasets\IEMOCAP\clips_by_label"
    train_dir = data_dir + r"\train"
    glob_pattern = train_dir+r"\{classname}\*.avi"
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    SIZE = (112, 112)
    IMAGE_CHANNELS = 3
    NBFRAME = 16
    optimizer = keras.optimizers.Adam(0.001)
    loss = "categorical_crossentropy"
    callbacks = [EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]

    print("---------------------------------------Prepering Data---------------------------------------")
    # Label Counter:  {'fru': 967, 'ang': 597, 'sad': 630, 'hap': 815}
    '''prep_data_folders("D:\Code\Colman\Research\Datasets\IEMOCAP\clips_by_label")
    classes = [i.split(os.path.sep)[1] for i in glob.glob(r"D:\Code\Colman\Research\Datasets\IEMOCAP\clips_by_label\train\*")]
    classes.sort()
    '''
    # for data augmentation
    data_aug = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.1,
        horizontal_flip=True,
        rotation_range=8,
        width_shift_range=.2,
        height_shift_range=.2)
    # Create video frame generator
    train = VideoFrameGenerator(
        classes=classes,
        glob_pattern=glob_pattern,
        nb_frames=NBFRAME,
        split=.1,
        shuffle=True,
        batch_size=BATCH_SIZE,
        target_shape=SIZE,
        nb_channel=IMAGE_CHANNELS,
        transformation=data_aug,
        use_frame_cache=True)
    valid = train.get_validation_generator()


    print("---------------------------------------Training Transfare Model---------------------------------------")
    transfare_model = C3D_transfare_model(weights_path="models\weights_C3D_sports1M_tf.h5", summary=True, trainable=False, num_layers_remove=10, num_classes=len(classes))
    transfare_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    transfare_model.fit(x=train, validation_data=valid, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS,
                        callbacks = callbacks, verbose = 1, shuffle = True)
    print("---------------------------------------Evaluate Transfare Model---------------------------------------")
    print("---------------------------------------Saving Transfare Model---------------------------------------")
    transfare_model.save("models/transfare_model_1.h5")

    print("---------------------------------------Training Ground Up Model---------------------------------------")
    groundup_model = C3D_groundup_model(len(classes))
    groundup_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    groundup_model.fit(x=train, validation_data=valid, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS,
                        callbacks = callbacks, verbose = 1, shuffle = True)
    print("---------------------------------------Evaluate Ground Up Model---------------------------------------")
    print("---------------------------------------Saving Ground Up Model---------------------------------------")
    groundup_model.save("models/groundup_model_1.h5")


    print("---------------------------------------DONE---------------------------------------")


if __name__ == "__main__":
    main()