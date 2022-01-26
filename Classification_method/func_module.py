import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ЗАГРУЖАЕМ МОДЕЛЬ
model_det = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model_det.eval()  # evaluation mode
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_cut_img(img):

    # Detection function
    def get_prediction(img, threshold=0.5):
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img).to(device)
        pred = model_det([img])
        pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        clock_indexes = [i for i in range(len(pred_score)) if (pred_score[i] > threshold and pred_classes[i] == 'clock')]
        pred_t = np.argmax([pred_score[i] for i in clock_indexes])    # best score clock
    #     pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_box = pred_boxes[clock_indexes[pred_t]]
        pred_class = pred_classes[clock_indexes[pred_t]]
        return pred_box, pred_class

    # Crop function
    def crop_img(img):
        pred = get_prediction(img)[0]
        (x1, y1), (x2, y2) = np.array(pred, dtype=int)
        crop_img = img[y1:y2, x1:x2]
        return crop_img

    crp = crop_img(img)

    return crp


def transform(img):
    # IMAGES TRANSFORMATION DEFINITION
    transform = transforms.Compose([
        transforms.ToTensor(),                                                  # преобразование в тензор
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),     # нормализация изображения
    ])

    img_t = transform(img) 

    return img_t


# CLASSIFICATION MODEL
def load_model():
    # LOAD AND FINETUNING PRETRAINED MODEL
    model_cls = models.resnext50_32x4d(pretrained=True)
    for param in model_cls.parameters():
        param.require = False
    num_ftrs = model_cls.fc.in_features
    model_cls.fc = nn.Linear(num_ftrs, num_ftrs)
    model_cls.fc = nn.Linear(num_ftrs, 720)

    # LOAD SAVED PARAMETERS
    param_file_name = 'model_classification.pth'  #model cls - epoch 2 of 50, train_loss 0.0070, valid_acc 1.0000.pth
    model_param = os.path.join(param_file_name)
    model_cls.load_state_dict(torch.load(model_param, map_location=device))

    return model_cls


# # REGRESSION MODEL
# def load_model():
#     # LOAD AND FINETUNING PRETRAINED MODEL
#     model_lr = models.resnext50_32x4d(pretrained=True)
#     for param in model_lr.parameters():
#         param.require = False
#     num_ftrs = model_cls.fc.in_features
#     model_lr.fc = nn.Linear(num_ftrs, num_ftrs)
#     model_lr.fc = nn.Linear(num_ftrs, 720)
#
#     # LOAD SAVED PARAMETERS
#     param_file_name = 'model_lr - epoch 29 of 30, train_loss 5.9186, valid_loss 17.7398.pth'
#     model_param = os.path.join('.', param_file_name)
#     model_lr.load_state_dict(torch.load(model_param))
#
#     return model_lr
#
#
# # OXFORD MODEL
# def load_model():
#     # LOAD AND FINETUNING PRETRAINED MODEL
#     model_cls = models.resnet50(pretrained=True)
#     num_ftrs = model_cls.fc.in_features
#     model_cls.fc = nn.Linear(num_ftrs, 720)
#
#     # LOAD SAVED PARAMETERS
#     param_file_name = 'full+++.pth'
#     model_param = os.path.join(param_file_name)
#     model_cls.load_state_dict(torch.load(model_param))
#
#     return model_cls
