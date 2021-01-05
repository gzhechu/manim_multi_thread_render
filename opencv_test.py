#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:04:25 2021

@author: hechu
"""
import cv2
import numpy as np
from tqdm import tqdm as show_progress


# TODO, is this depricated?
class SceneFromVideo():
    def read_file(self, file_name,
                  freeze_last_frame=True,
                  time_range=None):
        cap = cv2.VideoCapture(file_name)
        self.shape = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_rate = fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if time_range is None:
            start_frame = 0
            end_frame = frame_count
        else:
            start_frame, end_frame = [fps * t for t in time_range]

        frame_count = end_frame - start_frame
        print("Reading in " + file_name + "...")
        self.frames = []
        for count in show_progress(list(range(start_frame, end_frame + 1))):
            returned, frame = cap.read()
            if not returned:
                break
            # b, g, r = cv2.split(frame)
            # self.frames.append(cv2.merge([r, g, b]))
            self.frames.append(frame)
#            frames.append(frame)
        cap.release()

        if freeze_last_frame and len(self.frames) > 0:
            self.original_background = self.background = self.frames[-1]

    def apply_gaussian_blur(self, ksize=(5, 5), sigmaX=5):
        self.frames = [
            cv2.GaussianBlur(frame, ksize, sigmaX)
            for frame in self.frames
        ]

    def apply_edge_detection(self, threshold1=50, threshold2=100):
        edged_frames = [
            cv2.Canny(frame, threshold1, threshold2)
            for frame in self.frames
        ]
        for index in range(len(self.frames)):
            for i in range(3):
                self.frames[index][:, :, i] = edged_frames[index]

    def save_to_file(self, filename):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(filename, fourcc, self.frame_rate, self.shape)
        for frame in self.frames:
    #    ret, frame = cap.read()
            out.write(frame)

def test1():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#    fourcc = cv2.VideoWriter_fourcc(*'avc1')

#    frame = np.load('test/FRAME_1609755158_10.npy')
    frame = np.load('test/00012000275.npy')
#    frame = cv2.imread("test/test.png", cv2.IMREAD_COLOR)
#    frame = cv2.imread("test/test.png", cv2.IMREAD_UNCHANGED)
    frame = frame[:,:,:3]
    w = frame.shape[1]
    h = frame.shape[0]
    out = cv2.VideoWriter('output1.mp4', fourcc, 15.0, (w, h))

    frames = []
    for i in range (120):
        frames.append(frame)

    for frame in frames:
#    for i in range (120):
    #    ret, frame = cap.read()
        out.write(frame)
    #    cv2.imshow('frame', frame)
    #    c = cv2.waitKey(1)
    #    if c & 0xFF == ord('q'):
    #        break

#    cap.release()
    out.release()
    cv2.destroyAllWindows()


def test2():
    file_name = "/home/hechu/study/manim/media/videos/ex20201225_fixed_diff_value/640p15/FixedDiffValue2.mp4"
    vs = SceneFromVideo()
    vs.read_file(file_name)
    vs.apply_edge_detection()
#    vs.apply_gaussian_blur()
    vs.save_to_file("1.mp4")
    pass

test1()
