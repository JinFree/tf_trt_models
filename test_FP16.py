import cv2
import sys
import os
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf
import numpy as np
import time
from tf_trt_models.detection import download_detection_model, build_detection_graph

IMAGE_PATH = './COCO_val2014_000000581929.jpg'

def open_pb(PB_PATH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PB_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
 
trt_graph = open_pb('./data/faster_rcnn_resnet101_coco_trt_16.pb')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with trt_graph.as_default():
    tf_sess = tf.Session(config=tf_config)

    #tf.import_graph_def(trt_graph, name='')

    tf_input = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

    image_bgr = cv2.imread(IMAGE_PATH)
    image_origin = np.copy(image_bgr)
    image = image_bgr[..., ::-1]

    #image_resized = cv2.resize(image, (300, 300))

    scores, boxes, classes, num_detections = tf_sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None, ...]
    })

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = int(num_detections[0])

    rows = image_bgr.shape[0]
    cols = image_bgr.shape[1]
    # plot boxes exceeding score threshold
    for i in range(num_detections):
        classId = classes[i]
        score = scores[i]
        bbox = [float(v) for v in boxes[i]]
        if score > 0.1:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(image_bgr, (int(x), int(y)), (int(right), int(bottom)), (100, 255, 10), 5)

    cv2.imshow("origin", image_origin)
    cv2.imshow("result", image_bgr)
    cv2.waitKey()
