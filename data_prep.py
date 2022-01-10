import glob
import cv2
import numpy as np
import random
import face_recognition
import os
import shutil
from util import *
import tensorflow as tf
from tensorflow import keras
from keras_video import VideoFrameGenerator

def get_vid_frame_gen(data_dir, classes,batch_size):

    classes = classes
    data_dir = data_dir
    train_dir = data_dir + r"\train"
    glob_pattern = train_dir + r"\{classname}\*.avi"
    BATCH_SIZE = batch_size
    SIZE = (112, 112)
    IMAGE_CHANNELS = 3
    NBFRAME = 16

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
        transformation=None,
        use_frame_cache=False)
    valid = train.get_validation_generator()

    return train, valid


def prep_data_folders(path):
    os.chdir(path)
    if "train" in glob.glob("*"):
        print("Data is prepered")
        return

    count = {}
    for pic in glob.glob("*"):
        label = pic.split("\\")[-1].split(".")[0].split("-")[-1]
        if not label in count:
            count[label] = 1
        else:
            count[label] = count[label] + 1
    print("Label Counter: ",count)
    for pic in glob.glob("*xxx*"):
        print(f"Deleting: {pic}")
        os.remove(pic)

    for label in count:
        if count[label] < 10:
            for pic in glob.glob(f"*{label}*"):
                print(f"Deleting: {pic}")
                os.remove(pic)

        if os.path.isdir("train") is False:
            os.makedirs("train")
            os.makedirs("valid")

        if os.path.isdir(f"train/{label}") is False:
            os.makedirs(f"train/{label}")
            os.makedirs(f"valid/{label}")
            for pic in glob.glob(f"*{label}*"):
                shutil.move(pic, f"train/{label}")
    os.chdir("D:\Code\Colman\Research\Python\IEMOCAP_conv3d_video_classifier")

