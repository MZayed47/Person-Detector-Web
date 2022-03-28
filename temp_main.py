#! /usr/bin/env python
# coding=utf-8
from dis import dis
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5


# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5




from importlib.resources import path
import os
import cv2
import numpy as np
from glob import glob
import shutil

import random
import colorsys
from core.config import cfg
import re

from PIL import ImageGrab
from PIL import Image

import time
from time import gmtime, strftime
from datetime import datetime
import tensorflow as tf

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


import face_recognition
import csv


flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/video.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.70, 'iou threshold')
flags.DEFINE_float('score', 0.70, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within video')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')
flags.DEFINE_boolean('person', False, 'perform person detection')
flags.DEFINE_boolean('frames', False, 'get the frames with persons')
flags.DEFINE_boolean('identify', False, 'Identify the target person')




def main(got_name):
    print(got_name)

    def load_freeze_layer(model='yolov4', tiny=False):
        if tiny:
            return (
                ['conv2d_9', 'conv2d_12']
                if model == 'yolov3'
                else ['conv2d_17', 'conv2d_20']
            )

        elif model == 'yolov3':
            return ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            return ['conv2d_93', 'conv2d_101', 'conv2d_109']

    def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
        if is_tiny:
            if model_name == 'yolov3':
                layer_size = 13
                output_pos = [9, 12]
            else:
                layer_size = 21
                output_pos = [17, 20]
        elif model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
        with open(weights_file, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(layer_size):
                conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
                bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in output_pos:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in output_pos:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])


    def read_class_names(class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def load_config(FLAGS):
        STRIDES = np.array(__C.YOLO.STRIDES)
        ANCHORS = get_anchors(__C.YOLO.ANCHORS)
        XYSCALE = __C.YOLO.XYSCALE
        NUM_CLASS = len(read_class_names(__C.YOLO.CLASSES))

        return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

    def get_anchors(anchors_path, tiny=False):
        anchors = np.array(anchors_path)
        return anchors.reshape(2, 3, 2) if tiny else anchors.reshape(3, 3, 2)

    def image_preprocess(image, target_size, gt_boxes=None):
        ih, iw    = target_size
        h,  w, _  = image.shape

        scale = min(iw/w, ih/h)
        nw, nh  = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded

        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

    # helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    def format_boxes(bboxes, image_height, image_width):
        for box in bboxes:
            ymin = int(box[0] * image_height)
            xmin = int(box[1] * image_width)
            ymax = int(box[2] * image_height)
            xmax = int(box[3] * image_width)
            box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
        return bboxes

    def draw_bbox(image, bboxes, info = False, counted_classes = None, show_label=True, allowed_classes=list(read_class_names(__C.YOLO.CLASSES).values()), read_plate = False):
        classes = read_class_names(__C.YOLO.CLASSES)
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes):
            if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
            coor = out_boxes[i]
            score = out_scores[i]
            class_ind = int(out_classes[i])
            class_name = classes[class_ind]
            if class_name not in allowed_classes:
                continue
            if read_plate:
                height_ratio = int(image_h / 25)
                plate_number = utils.recognize_plate(image, coor)
                if plate_number != None:
                    cv2.putText(image, plate_number, (int(coor[0]), int(coor[1]-height_ratio)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,0), 2)

            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if info:
                print("Object found: {}, Confidence: {:.2f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))

            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                fontScale = 0.5
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

            if counted_classes != None:
                height_ratio = int(image_h / 25)
                offset = 15
                for key, value in counted_classes.items():
                    cv2.putText(image, "{}s detected: {}".format(key, value), (5, offset),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                    offset += height_ratio
        return image

    def bbox_iou(bboxes1, bboxes2):
        """
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        return tf.math.divide_no_nan(inter_area, union_area)


    def bbox_giou(bboxes1, bboxes2):
        """
        Generalized IoU
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        enclose_right_down = tf.maximum(
            bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
        )

        enclose_section = enclose_right_down - enclose_left_up
        enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

        return iou - tf.math.divide_no_nan(enclose_area - union_area, enclose_area)


    def bbox_ciou(bboxes1, bboxes2):
        """
        Complete IoU
        @param bboxes1: (a, b, ..., 4)
        @param bboxes2: (A, B, ..., 4)
            x:X is 1:n or n:n or n:1
        @return (max(a,A), max(b,B), ...)
        ex) (4,):(3,4) -> (3,)
            (2,1,4):(2,3,4) -> (2,3)
        """
        bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
        bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

        bboxes1_coor = tf.concat(
            [
                bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
                bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
            ],
            axis=-1,
        )
        bboxes2_coor = tf.concat(
            [
                bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
                bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
            ],
            axis=-1,
        )

        left_up = tf.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        right_down = tf.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        union_area = bboxes1_area + bboxes2_area - inter_area

        iou = tf.math.divide_no_nan(inter_area, union_area)

        enclose_left_up = tf.minimum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
        enclose_right_down = tf.maximum(
            bboxes1_coor[..., 2:], bboxes2_coor[..., 2:]
        )

        enclose_section = enclose_right_down - enclose_left_up

        c_2 = enclose_section[..., 0] ** 2 + enclose_section[..., 1] ** 2

        center_diagonal = bboxes2[..., :2] - bboxes1[..., :2]

        rho_2 = center_diagonal[..., 0] ** 2 + center_diagonal[..., 1] ** 2

        diou = iou - tf.math.divide_no_nan(rho_2, c_2)

        v = (
            (
                tf.math.atan(
                    tf.math.divide_no_nan(bboxes1[..., 2], bboxes1[..., 3])
                )
                - tf.math.atan(
                    tf.math.divide_no_nan(bboxes2[..., 2], bboxes2[..., 3])
                )
            )
            * 2
            / np.pi
        ) ** 2

        alpha = tf.math.divide_no_nan(v, 1 - iou + v)

        return diou - alpha * v

        # return ciou

    def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes


    def freeze_all(model, frozen=True):
        model.trainable = not frozen
        if isinstance(model, tf.keras.Model):
            for l in model.layers:
                freeze_all(l, frozen)


    def unfreeze_all(model, frozen=False):
        model.trainable = not frozen
        if isinstance(model, tf.keras.Model):
            for l in model.layers:
                unfreeze_all(l, frozen)


###############################################################################
###############################################################################

    yy = strftime("%d-%b-%Y_%H-%M", gmtime())
    # print(yy)

    # Source Images
    images = []
    classNames = []


    path = 'static/people'
    name = got_name
    cl = name + ".jpg"
    # print(cl)
    curImg = cv2.imread(f'{path}/{cl}')
    # print(cl)
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

    # Face Encodings
    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(images)
    # print('Encoding Complete')
    # print("Number of Records: ",len(encodeListKnown))


    ###############################################################

    FLAGS.weights = './checkpoints/yolov4-416'
    FLAGS.model = 'yolov4'
    FLAGS.size = 416


    # For only video
    FLAGS.video = 'uploaded_video.mp4'
    FLAGS.output = "./static/detections/room_output.mp4"

    # FLAGS.person = True
    FLAGS.crop = True
    FLAGS.frames = True
    FLAGS.identify = True

    person_count = 0

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = load_config(FLAGS)
    input_size = 416
    video_path = 'uploaded_video.mp4'
    # get video name by using split method
    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]


    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None


    if "./static/detections/room_output.mp4":
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        vid_fps = fps
        # print(vid_fps)
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("./static/detections/room_output.mp4", codec, fps, (width, height))

    ##############################################################################
    # Main Loop Starts
    frame_num = -1

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('\n--- Video has ended or failed. Check the output or try a different video format! ---')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=0.25
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = read_class_names(__C.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        allowed_classes = ['person']


        # if crop flag is enabled, crop each detection and save it as new image
        crop_rate = int(vid_fps/2) # capture images every so many frames (ex. crop photos every 150 frames)
        crop_path = os.path.join(os.getcwd(), 'static', 'detections', 'crop_' + yy)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        if frame_num % crop_rate == 0:
            final_path = os.path.join(crop_path, 'frame_' + str(frame_num))
            try:
                os.mkdir(final_path)
            except FileExistsError:
                pass
            crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes)
        else:
            pass


        for file in glob("./static/detections/crop_" + yy + "/*/", recursive = True):
            # Get the file names
            ff = os.path.normpath(file)
            xx = os.path.basename(ff)

            # Get all the images in the folders
            for i in os.listdir(file):
                # print(i)
                ii = './static/detections/crop_' + yy + '/' + xx + '/' + i
                image_i = cv2.imread(ii)
                cv2.imwrite('./static/detections/crop_' + yy + '/' + xx + '_' + i, image_i)
            shutil.rmtree(file)


        file = "./static/detections/crop_" + yy + "/"

        for i in os.listdir(file):
            img_name = i.split('.')[0]
            # print(img_name)
            try:
                frame_i = cv2.imread("./static/detections/crop_" + yy + "/" + i)
                frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
                facesCurFrame = face_recognition.face_locations(frame_i)
                encodesCurFrame = face_recognition.face_encodings(frame_i, facesCurFrame)

                person_path = os.path.join(os.getcwd(), 'static', 'detections', 'person_' + yy)
                isdir = os.path.isdir(person_path)
                if not isdir:
                    os.mkdir(person_path)

                for encodeFace,faceLoc in zip(encodesCurFrame, facesCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    # print(faceDis)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()
                        # print(name, 'is present in the video.\n')

                        y1,x2,y2,x1 = faceLoc
                        # y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(frame_i,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.rectangle(frame_i,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(frame_i, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)
                        
                        frame_i = cv2.cvtColor(frame_i, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(person_path + '/' + img_name + '_' + name + '.jpg', frame_i)

                        person_count += 1

            except:
                pass


        image = draw_bbox(frame, pred_bbox, False, allowed_classes=allowed_classes, read_plate=False)



        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(image)
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    big_line = []
    if person_count != 0:
        out_image = cl
        big_line.append(out_image)
        print(out_image)

        display_output = "\n\n--- Result: Positive! " + name + " is present in the video. ---"
        big_line.append(display_output)
        print('\n\n--- Result: Positive! ', name, 'is present in the video. ---\n')

        got_frames = "./static/detections/person_" + yy + "/"
        frame_path = 'person_' + yy
        big_line.append(frame_path)

        print('The captured frames are:')
        for f in os.listdir(got_frames):
            fr_name = f.split('.')[0]
            print(fr_name)
            big_line.append(f)
        print()

    else:
        out_image = cl
        big_line.append(out_image)
        print(out_image)

        display_output = "\n\n--- Result: Negative! " + name + " is not present in the video. ---"
        big_line.append(display_output)
        print('\n\n--- Result: Negative! ', name, 'is not present in the video. ---\n\n')


    cv2.destroyAllWindows()

    return big_line


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
