"""
Yolo v2 loss function.
"""
from src.object_detection.constants import *
import numpy as np
import tensorflow as tf

def generate_yolo_grid(batch, g_h, g_w, num_bb):
    c_x = tf.keras.backend.cast(tf.keras.backend.reshape(tf.keras.backend.tile(tf.keras.backend.arange(g_h), [g_w]), (1, g_h, g_w, 1, 1)), tf.keras.backend.floatx())
    c_y = tf.keras.backend.permute_dimensions(c_x, (0,2,1,3,4))
    return tf.keras.backend.tile(tf.keras.backend.concatenate([c_x, c_y], -1), [batch, 1, 1, num_bb, 1])


def calculate_ious(A1, A2, use_iou=True):

    if not use_iou:
        return A1[..., 4]

    def process_boxes(A):
        # ALign x-w, y-h
        A_xy = A[..., 0:2]
        A_wh = A[..., 2:4]
        
        A_wh_half = A_wh / 2.
        # Get x_min, y_min
        A_mins = A_xy - A_wh_half
        # Get x_max, y_max
        A_maxes = A_xy + A_wh_half
        
        return A_mins, A_maxes, A_wh
    
    # Process two sets
    A2_mins, A2_maxes, A2_wh = process_boxes(A2)
    A1_mins, A1_maxes, A1_wh = process_boxes(A1)

    # Intersection as min(Upper1, Upper2) - max(Lower1, Lower2)
    intersect_mins  = tf.keras.backend.maximum(A2_mins,  A1_mins)
    intersect_maxes = tf.keras.backend.minimum(A2_maxes, A1_maxes)
    
    # Getting the intersections in the xy (aka the width, height intersection)
    intersect_wh = tf.keras.backend.maximum(intersect_maxes - intersect_mins, 0.)

    # Multiply to get intersecting area
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Values for the single sets
    true_areas = A1_wh[..., 0] * A1_wh[..., 1]
    pred_areas = A2_wh[..., 0] * A2_wh[..., 1]

    # Compute union for the IoU
    union_areas = pred_areas + true_areas - intersect_areas
    return intersect_areas / union_areas


def _transform_netout(y_pred):
    coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
    coord_y = tf.transpose(coord_x, (0,2,1,3,4))
    coords = tf.tile(tf.concat([coord_x,coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])
    dims = tf.keras.backend.cast_to_floatx(tf.keras.backend.int_shape(y_pred)[1:3])
    dims = tf.keras.backend.reshape(dims,(1,1,1,1,2))
    # anchors tensor
    anchors = np.array(ANCHORS)
    anchors = anchors.reshape(len(anchors) // 2, 2)
    # pred_xy and pred_wh shape (m, GRID_W, GRID_H, Anchors, 2)
    pred_xy = tf.keras.backend.sigmoid(y_pred[:,:,:,:,0:2])
    pred_xy = (pred_xy + coords)
    pred_xy = pred_xy / dims
    pred_wh = tf.keras.backend.exp(y_pred[:,:,:,:,2:4])
    pred_wh = (pred_wh * anchors)
    pred_wh = pred_wh / dims
    # pred_confidence
    box_conf = tf.keras.backend.sigmoid(y_pred[:,:,:,:,4:5])
    # pred_class
    box_class_prob = tf.keras.backend.softmax(y_pred[:,:,:,:,5:])
    """
    print(pred_xy.shape)
    print(pred_wh.shape)
    print(box_conf.shape)
    print(box_class_prob.shape)
    """
    return tf.keras.backend.concatenate([pred_xy, pred_wh, box_conf, box_class_prob], axis=-1)

class YoloLoss(object):

    def __init__(self):
        self.__name__ = 'yolo_loss'
        self.iou_filter = 0.6
        self.readjust_obj_score = False

        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJECT
        self.lambda_obj = LAMBDA_OBJECT
        self.lambda_class = LAMBDA_CLASS


    def coord_loss(self, y_true, y_pred):
        pred_xy = y_pred[:,:,:,:,0:2]
        pred_wh = y_pred[:,:,:,:,2:4]
        detector_mask = tf.expand_dims(y_pred[:,:,:,:,4], -1)
        nb_detector_mask = tf.keras.backend.sum(tf.cast(detector_mask > 0.0, tf.float32))
        xy_loss = self.lambda_coord * tf.keras.backend.sum(
            detector_mask * tf.keras.backend.square(y_true[...,:2] - pred_xy)
        ) / (nb_detector_mask + EPSILON)
        wh_loss = self.lambda_coord * tf.keras.backend.sum(
            detector_mask * tf.keras.backend.square(tf.keras.backend.sqrt(y_true[...,2:4]) - tf.keras.backend.sqrt(pred_wh))
        ) / (nb_detector_mask + EPSILON)

        return xy_loss + wh_loss


    def obj_loss(self, y_true, y_pred):

        b_o = calculate_ious(y_true, y_pred, use_iou=self.readjust_obj_score)
        b_o_pred = y_pred[..., 4]

        num_true_labels = GRID_H*GRID_W*BOX
        y_true_p = tf.keras.backend.reshape(y_true[..., :4], shape=(y_true.shape[0], 1, 1, 1, num_true_labels, 4))
        iou_scores_buff = calculate_ious(y_true_p, tf.keras.backend.expand_dims(y_pred, axis=4))
        best_ious = tf.keras.backend.max(iou_scores_buff, axis=4)

        indicator_noobj = tf.keras.backend.cast(best_ious < self.iou_filter, np.float32) * (1 - y_true[..., 4]) * self.lambda_noobj
        indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_o = indicator_obj + indicator_noobj

        loss_obj = tf.keras.backend.sum(tf.keras.backend.square(b_o-b_o_pred) * indicator_o)#, axis=[1,2,3])

        return loss_obj / 2


    def class_loss(self, y_true, y_pred):
        loss_class = tf.keras.backend.sum(tf.keras.backend.categorical_crossentropy(y_true[..., 5:], y_pred[..., 5:]))
        return loss_class
    
    def l_coord(self, y_true, y_pred_raw):
        return self.coord_loss(y_true, _transform_netout(y_pred_raw))

    def l_obj(self, y_true, y_pred_raw):
        return self.obj_loss(y_true, _transform_netout(y_pred_raw))

    def l_class(self, y_true, y_pred_raw):
        return self.class_loss(y_true, _transform_netout(y_pred_raw))

    def __call__(self, y_true, y_pred_raw, info = False):

        y_pred = _transform_netout(y_pred_raw)

        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)
        sub_loss = [total_obj_loss, total_class_loss, total_coord_loss]
        loss = total_coord_loss + total_obj_loss + total_class_loss

        if info:
            print('conf_loss   : {:.4f}'.format(total_obj_loss))
            print('class_loss  : {:.4f}'.format(total_class_loss))
            print('coord_loss  : {:.4f}'.format(total_coord_loss))
            print('--------------------')
            print('total loss  : {:.4f}'.format(loss))

        return  loss, sub_loss

LOSS = YoloLoss()
