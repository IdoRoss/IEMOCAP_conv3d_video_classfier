import glob
import cv2
import numpy as np
import random
import face_recognition
import os
import shutil
from util import *



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

