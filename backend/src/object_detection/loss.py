"""
Yolo v2 loss function.
"""
from constants import *
import numpy as np
import tensorflow as tf

EPSILON = 1e-7


def generate_yolo_grid(batch, g, num_bb):
    c_x = K.cast(K.reshape(K.tile(K.arange(g), [g]), (1, g, g, 1, 1)), K.floatx())
    c_y = K.permute_dimensions(c_x, (0,2,1,3,4))
    return K.tile(K.concatenate([c_x, c_y], -1), [batch, 1, 1, num_bb, 1])

c_grid = generate_yolo_grid(BATCH_SIZE, GRID_H, GRID_W, BOX)
anchors = np.reshape(ANCHORS, [1, 1, 1, BOX, 2])

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


def _transform_netout(y_pred_raw):
    y_pred_xy = tf.keras.backend.sigmoid(y_pred_raw[..., :2]) + c_grid
    y_pred_wh = tf.keras.backend.exp(y_pred_raw[..., 2:4]) * anchors
    y_pred_conf = tf.keras.backend.sigmoid(y_pred_raw[..., 4:5])
    y_pred_class = y_pred_raw[...,5:]

    return tf.keras.backend.concatenate([y_pred_xy, y_pred_wh, y_pred_conf, y_pred_class], axis=-1)

class YoloLoss(object):

    def __init__(self, batch_size):
        self.__name__ = 'yolo_loss'
        self.iou_filter = 0.6
        self.readjust_obj_score = False

        self.lambda_coord = LAMBDA_COORD
        self.lambda_noobj = LAMBDA_NOOBJECT
        self.lambda_obj = LAMBDA_OBJECT
        self.lambda_class = LAMBDA_CLASS


    def coord_loss(self, y_true, y_pred):
        
        b_xy_pred = y_pred[..., :2]
        b_wh_pred = y_pred[..., 2:4]
        
        b_xy = y_true[..., 0:2]
        b_wh = y_true[..., 2:4]

        indicator_coord = tf.keras.backend.expand_dims(y_true[..., 4], axis=-1) * self.lambda_coord

        loss_xy = tf.keras.backend.sum(tf.keras.backend.square(b_xy - b_xy_pred) * indicator_coord)#, axis=[1,2,3,4])
        loss_wh = tf.keras.backend.sum(tf.keras.backend.square(b_wh - b_wh_pred) * indicator_coord)#, axis=[1,2,3,4])
        #loss_wh = tf.keras.backend.sum(tf.keras.backend.square(tf.keras.backend.sqrt(b_wh) - tf.keras.backend.sqrt(b_wh_pred)) * indicator_coord)#, axis=[1,2,3,4])

        return (loss_wh + loss_xy) / 2


    def obj_loss(self, y_true, y_pred):

        b_o = calculate_ious(y_true, y_pred, use_iou=self.readjust_obj_score)
        b_o_pred = y_pred[..., 4]

        num_true_labels = GRID_H*GRID_W*BOX
        y_true_p = tf.keras.backend.reshape(y_true[..., :4], shape=(BATCH_SIZE, 1, 1, 1, num_true_labels, 4))
        iou_scores_buff = calculate_ious(y_true_p, tf.keras.backend.expand_dims(y_pred, axis=4))
        best_ious = tf.keras.backend.max(iou_scores_buff, axis=4)

        indicator_noobj = tf.keras.backend.cast(best_ious < self.iou_filter, np.float32) * (1 - y_true[..., 4]) * self.lambda_noobj
        indicator_obj = y_true[..., 4] * self.lambda_obj
        indicator_o = indicator_obj + indicator_noobj

        loss_obj = tf.keras.backend.sum(tf.keras.backend.square(b_o-b_o_pred) * indicator_o)#, axis=[1,2,3])

        return loss_obj / 2


    def class_loss(self, y_true, y_pred):

        p_c_pred = tf.keras.backend.softmax(y_pred[..., 5:])
        p_c = tf.keras.backend.one_hot(tf.keras.backend.argmax(y_true[..., 5:], axis=-1), CLASS)
        loss_class_arg = tf.keras.backend.sum(tf.keras.backend.square(p_c - p_c_pred), axis=-1)
        
        #b_class = tf.keras.backend.argmax(y_true[..., 5:], axis=-1)
        #b_class_pred = y_pred[..., 5:]
        #oss_class_arg = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=b_class, logits=b_class_pred)

        indicator_class = y_true[..., 4] * self.lambda_class

        loss_class = tf.keras.backend.sum(loss_class_arg * indicator_class)#, axis=[1,2,3])

        return loss_class
    
    def l_coord(self, y_true, y_pred_raw):
        return self.coord_loss(y_true, _transform_netout(y_pred_raw))

    def l_obj(self, y_true, y_pred_raw):
        return self.obj_loss(y_true, _transform_netout(y_pred_raw))

    def l_class(self, y_true, y_pred_raw):
        return self.class_loss(y_true, _transform_netout(y_pred_raw))



    def __call__(self, y_true, y_pred_raw):

        y_pred = _transform_netout(y_pred_raw)

        total_coord_loss = self.coord_loss(y_true, y_pred)
        total_obj_loss = self.obj_loss(y_true, y_pred)
        total_class_loss = self.class_loss(y_true, y_pred)

        loss = total_coord_loss + total_obj_loss + total_class_loss

        return  loss
