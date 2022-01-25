# :alarm_clock: :clock6: :watch: Clock_Project
The main question was - how to determine time from a photo of an analog clock using neural networks.

# Project description
2 options for solving the problem were worked out:
1. Trigonometric method:

segmentation of clock hands, detection of dial numbers and time determination through the use of the angle of the hands.

2. ResNext50 classification:

training ResNext50 on a dataset of over 200,000 images (720 image classes). 

# Project stages
## Trigonometric method:

1) annotate the dataset:
- for arrow segmentation (https://www.makesense.ai/)
- for detection of dial numbers (https://roboflow.com/)
2) training and obtaining model weights:
- mask_rcnn - for arrow segmentation
- faster_rcnn - for number detection
3) getting coordinates of arrows and numbers
4) determining time with coordinates and trigonometric functions

(!) Due to the low accuracy, it was decided to change the approach of time determination.

## ResNext50 classification:

1) creating a dataset with timelaps video split
2) creating a dataset with the clock image generation function
3) training the ResNext50 model on a generalized dataset 

# Results
Inference on ResNext50 gives 60% accuracy.

Telegram bot ... became the final product of the project.

## :man: :computer: Co-authors:

https://github.com/tano4ku

https://github.com/AdelGR

https://github.com/PavelBogoslovskiy
