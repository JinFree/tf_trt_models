import cv2
import sys
import os
import urllib
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
from tf_trt_models.classification import download_classification_checkpoint, build_classification_graph

PB32_PATH = './SE-ResNeXt.pb'
CHECKPOINT_PATH = './ckpt_n_pb/SE_ResNeXt_epoch_26.ckpt'
NUM_CLASSES = 1000
LABELS_PATH = './imagenet_labels.txt'
output_node_names = ['Softmax']
input_node_names = ['Placeholder']

def open_pb(PB_PATH):
    graph_def = tf.GraphDef()
    graph = tf.Graph()
    with tf.gfile.GFile(PB_PATH, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
    return graph, graph_def
    

_, frozen_graph = open_pb(PB32_PATH)
input_names = input_node_names
output_names = output_node_names

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

with open('./SE-ResNeXt_16.pb', 'wb') as f:
    f.write(trt_graph.SerializeToString())

print('FP16 변환 완료')
