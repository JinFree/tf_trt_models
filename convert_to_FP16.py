import cv2
import sys
import os
import urllib
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph

IMAGE_PATH = './COCO_val2014_000000581929.jpg'

## 아래 세 줄만 수정하면 실제 훈련한 모델에서도 활용 가능

PATH_TO_FASTER_RCNN = '/home/ldcc-xavier2/EdgeComputingPoC/Detection/faster_rcnn_resnet101_coco_2018_01_28/'
config_path = PATH_TO_FASTER_RCNN + 'pipeline.config'
checkpoint_path = PATH_TO_FASTER_RCNN + 'model.ckpt' 

frozen_graph, input_names, output_names = build_detection_graph(
    config=config_path,
    checkpoint=checkpoint_path,
    score_threshold=0.3,
    batch_size=1
)

print(output_names)

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

with open('./data/faster_rcnn_resnet101_coco_trt_16.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())

print('FP16 변환 완료')
