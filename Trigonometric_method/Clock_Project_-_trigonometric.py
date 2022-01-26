from imports import import_lib
from functions import best_detect_num, trr, len_line, arr, calculate_angle, mask, coord_hm, angle_12_3, time 

# предварительно нужно установить:
# pip install pyyaml==5.1
# pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
# pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

# все импорты
import_lib()

# берём предобученные веса faster_rcnn_R_50_FPN_3x
cfg_crop = get_cfg()
cfg_crop.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg_crop.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
cfg_crop.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictors_clock  = DefaultPredictor(cfg_crop)

# берём НАШИ веса для сегментации стрелок
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = '/models/model_segmentation.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
predictor_arrow = DefaultPredictor(cfg)

# берём НАШИ веса для детекции цифр
cfg_time = get_cfg()
cfg_time.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
cfg_time.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg_time.MODEL.WEIGHTS = '/models/model_detection.pth'
cfg_time.MODEL.ROI_HEADS.NUM_CLASSES = 13
predictor_time = DefaultPredictor(cfg_time)

# т.к. модель "переставляет" классы местами, указываем верный порядок:
trans_class = {1: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 6, 2: 7, 3: 8, 4: 9, 5: 10, 6: 11, 7: 0}

# определяем время:
im = cv2.imread('../photo_clock.jpg')
outputs_clock = predictors_clock(im)
clock_class = outputs_clock['instances'].pred_classes.cpu().numpy()
clock_box = outputs_clock['instances'].pred_boxes.tensor.cpu().numpy()
for i in range(len(clock_class)):
    if clock_class[i] == 74:
        box = clock_box[i]
        break

im = Image.fromarray(im)
im_crop = im.crop(box)
im_crop.save('crop.jpg', quality=95)

im = cv2.imread('crop.jpg')

outputs_arrow = predictor_arrow(im)
outputs_num = predictor_time(im)
h, m = time(im, outputs_arrow, outputs_num)
cv2_imshow(im)
print('время ' + str(h) + ':' + str(m))
