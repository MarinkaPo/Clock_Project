def import_lib():
    import torch, torchvision
    assert torch.__version__.startswith("1.9") 
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    import numpy as np
    import os, json, cv2, random
    import math
    import pandas as pd
    from google.colab.patches import cv2_imshow
    import matplotlib as plt
    import random
    import copy
    from PIL import Image
    from tqdm import tqdm

    from detectron2 import model_zoo
    from detectron2.engine import DefaultTrainer
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import COCOEvaluator
    from detectron2.config import get_cfg
    from detectron2.data import detection_utils as utils
    from detectron2.data import detection_utils as utils
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data import build_detection_test_loader, build_detection_train_loader
    from detectron2.data import transforms as T
    from detectron2.data.datasets import register_coco_instances
    from detectron2.utils.visualizer import Visualizer