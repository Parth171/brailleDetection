'''
title: Braille Detection
title: 7/26/2023
author: Parth Sakpal
'''

from ultralytics import YOLO
import numpy

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/User/PycharmProjects/brailleDetection/data/braille_data', epochs=1, imgsz=64)

