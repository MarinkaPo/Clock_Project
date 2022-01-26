import os
# import numpy as np
import cv2
import torch
# import torch.nn as nn
import torchvision.models as models
# import torchvision.transforms as transforms
from func_module import detect_cut_img, transform, load_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predictor(img_path):

    # READ IMG
    img = cv2.imread(img_path)

    # DETECTION + TRANSFORMATION
    img_c = detect_cut_img(img)
    img_c = cv2.resize(img_c, (224, 224))/255  # resize
    img_t = transform(img_c)                   # to tensor + normalize
    img_t = img_t.to(torch.float32)            # to float32
    img_t = torch.unsqueeze(img_t, 0)          # expand dimension

    # LOAD CLS MODEL
    model_cls = load_model()
    model_cls.to(device)
    model_cls.eval()

    # PREDICTION
    output = model_cls(img_t.to(device))
    _, prediction = torch.max(output.data, 1)
    prediction = prediction.item()

    # PREDICTION CONVERT
    hours = prediction // 60
    minutes = prediction % 60
    # time = f'{hours} hours, {minutes} minutes'

    if len(str(hours)) == 1: hours = '0' + str(hours)
    if len(str(minutes)) == 1: minutes = '0' + str(minutes)

    time = f'{hours}:{minutes}'
    
    return time







