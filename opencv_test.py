#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 17:04:25 2021

@author: hechu
"""
import cv2
import numpy as np

f = np.load('test/FRAME_1609755158_10.npy')


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 15.0, (640,480))

frame = cv2.imread("test/test.png", cv2.IMREAD_COLOR)

for i in range (100):
#    ret, frame = cap.read()
    out.write(frame)
#    cv2.imshow('frame', frame)
#    c = cv2.waitKey(1)
#    if c & 0xFF == ord('q'):
#        break

cap.release()
out.release()
cv2.destroyAllWindows()