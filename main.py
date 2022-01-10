from data_prep import *
from c3dmodel import *
from test_model import *
from train_model import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from keras.models import load_model
from keras_video import VideoFrameGenerator

# TODO: recrop clips or frames to reduce noise
# TODO: figure out how to use predict
def main():
    # use GPU
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    print("Num GPUs Available: ", len(physical_devices))
    if len(physical_devices) != 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


    classes = ['fru', 'ang', 'sad', 'hap']
    data_dir = "D:\Code\Colman\Research\Datasets\IEMOCAP\clips_by_label"
    NUM_EPOCHS = 100
    BATCH_SIZE = 8


    # Label Counter:  {'fru': 967, 'ang': 597, 'sad': 630, 'hap': 815}
    train, valid = get_vid_frame_gen(data_dir, classes, BATCH_SIZE)

    transfare_model = C3D_transfare_model(weights_path="models\weights_C3D_sports1M_tf.h5", summary=True, num_classes=len(classes))
    train_c3d_model(transfare_model, "transfare_model1", NUM_EPOCHS, train, valid)
    test_model(transfare_model, valid, classes)

    # groundup_model = C3D_groundup_model(len(classes))


if __name__ == "__main__":
    main()

    '''import keras_video.utils
    keras_video.utils.show_sample(train)
    test_model(model=load_model("models/groundup_model_1.h5"),batches= train,classes=classes)
    test_model(model=load_model("models/transfare_model_1.h5"), batches=train,classes=classes)'''