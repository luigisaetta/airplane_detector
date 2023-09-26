#!/usr/bin/env python
# coding: utf-8

# ### YOLO V8 Plane Detector
# * using directly YOLO v8 model
# * process a video

# In[1]:


import math
from os import path
from tqdm import tqdm
import cv2
import argparse

from ultralytics import YOLO


#
# Functions
#
def parser_add_args(parser):
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Input video name",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="YOLO v8 model path",
    )
    return parser


#
# Main
#
# command line parms
parser = argparse.ArgumentParser()
parser = parser_add_args(parser)
args = parser.parse_args()

print()
print("Executing with:")
print(vars(args))
print()

# Video in input
VIDEO_PATH = args.input_video

# load model
MODEL_PATH = args.model_path

# passati a modello large
# MODEL_PATH = "./runs/detect/train8/weights/best.pt"

model = YOLO(MODEL_PATH)

# set model parameters
# model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides["iou"] = 0.45  # NMS IoU threshold
# model.overrides['agnostic_nms'] = False  # NMS class-agnostic
# model.overrides['max_det'] = 1000


# In[3]:


# VIDEO_PATH = "mixkit-plane-taking-off.mp4"
CODEC = "mp4v"

ONLY_NAME = VIDEO_PATH.split(".")[-2]
# remove /
ONLY_NAME = ONLY_NAME.replace("/", "")

# name of the video with BB
VIDEO_OUT = f"{ONLY_NAME}_bb.mp4"

# settings for the BB that will be added
# remember: OpenCV is BGR
color = (0, 0, 255)  # red
tickness = 1


# for the original video
cap = cv2.VideoCapture(VIDEO_PATH)

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))

print("Analyzed input video:")
print(f"Number of frames is: {n_frames}")
print(f"Fps are: {fps}")
print()


# #### Apply YOLO to single frames

# In[6]:


print(f"Processing input video {VIDEO_PATH}...")
print()

# read the first frame and initialize
ret, frame = cap.read()

# take height, width from the first frame
height, width, layers = frame.shape

# the annotated (output) video
video = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*CODEC), fps, (width, height))

# process frame by frame
with tqdm(total=n_frames) as pbar:
    while ret:
        # apply yolo to the single frame
        results = model.predict(frame, verbose=False)

        # make a copy to add bb
        new_image = frame.copy()

        for result in results:
            # move to cpu
            result = result.cpu()
            # cast to int
            boxes = result.boxes.xyxy.numpy().astype(int)

            for box in boxes:
                # add rectangle
                new_image = cv2.rectangle(
                    new_image, (box[0], box[1]), (box[2], box[3]), color, tickness
                )

        # write the annotated image in the video
        video.write(new_image)

        # update the progress bar
        pbar.update(1)

        # next frame
        ret, frame = cap.read()

# close the output
video.release()

# close the input
cap.release()


# In[8]:


print()
print("Process video with YOLO v8 model correctly terminated.")
print(f"Output file produced: {VIDEO_OUT}")
print()


# In[ ]:
