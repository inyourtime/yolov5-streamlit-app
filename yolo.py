import torch
import numpy as np
import cv2
import time

def load_model(model_type='yolov5s'):
    typef = f"{model_type}.pt"
    return torch.hub.load('ultralytics/yolov5', 'custom', path=typef)

def model_config(model, conf, iou):
    model.conf = conf
    model.iou = iou

    return model

def class_to_label(model, x):
    return model.names[int(x)]

def score_frame(model, frame):
    model.to('cuda')
    frame = [frame]
    
    results = model(frame, size=640)

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates, results

def plot_boxes(results, frame, conf, model):
    labels, cord, _ = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= conf:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (220, 88, 42)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
            text = class_to_label(model, labels[i]) + " " + str(round(row[4].item(), 2))
            draw_text(frame, text, pos=(x1,y1), text_color_bg=bgr)
    
    return frame

def draw_text(img, text,
            font=cv2.FONT_HERSHEY_PLAIN,
            pos=(0,0),
            font_scale=3,
            font_thickness=2,
            text_color=(255, 255, 255),
            text_color_bg=(0, 0, 0)
            ):
    x, y = pos
    text_w, text_h = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    cv2.rectangle(img, (x, y - text_h), (x + text_w, y), text_color_bg, -1)
    cv2.putText(img, text, (x, y + font_scale - 1), font, font_scale, text_color, font_thickness)