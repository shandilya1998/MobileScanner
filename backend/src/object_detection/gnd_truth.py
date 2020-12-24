from src.object_detection.constants import *
import tensorflow as tf
import numpy as np

"""## 1.0. Process data to YOLO prediction format"""

def process_true_boxes(true_boxes, anchors, image_width, image_height):
    '''
    Build image ground truth in YOLO format from image true_boxes and anchors.
    
    Parameters
    ----------
    - true_boxes : tensor, shape (max_annot, 5), format : x1 y1 x2 y2 c, coords unit : image pixel
    - anchors : list [anchor_1_width, anchor_1_height, anchor_2_width, anchor_2_height...]
        anchors coords unit : grid cell
    - image_width, image_height : int (pixels)
    
    Returns
    -------
    - detector_mask : array, shape (GRID_H, GRID_W, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : array, shape (GRID_H, GRID_W, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    -true_boxes_grid : array, same shape than true_boxes (max_annot, 5),
        format : x, y, w, h, c, coords unit : grid cell
        
    Note:
    -----
    Bounding box in YOLO Format : x, y, w, h, c
    x, y : center of bounding box, unit : grid cell
    w, h : width and height of bounding box, unit : grid cell
    c : label index
    '''
    
    scale_w = IMAGE_W / GRID_W # scale = 32
    scale_h = IMAGE_H / GRID_H

    anchors_count = len(anchors) // 2
    anchors = np.array(anchors)
    anchors = anchors.reshape(len(anchors) // 2, 2)
    
    detector_mask = np.zeros((GRID_H, GRID_W, anchors_count, 1))
    matching_true_boxes = np.zeros((GRID_H, GRID_W, anchors_count, 5))
    
    # convert true_boxes numpy array -> tensor
    true_boxes = true_boxes.numpy()
    
    true_boxes_grid = np.zeros(true_boxes.shape)
    
    # convert bounding box coords and localize bounding box
    for i, box in enumerate(true_boxes):
        # convert box coords to x, y, w, h and convert to grids coord
        w = (box[2] - box[0]) / scale_w
        h = (box[3] - box[1]) / scale_h
        x = ((box[0] + box[2]) / 2) / scale_w
        y = ((box[1] + box[3]) / 2) / scale_h
        true_boxes_grid[i,...] = np.array([y, x, h, w, box[4]])
        if w * h > 0: # box exists
            # calculate iou between box and each anchors and find best anchors
            best_iou = 0
            best_anchor = 0
            for i in range(anchors_count):
                # iou (anchor and box are shifted to 0,0)
                intersect = np.minimum(w, anchors[i,0]) * np.minimum(h, anchors[i,1])
                union = (anchors[i,0] * anchors[i,1]) + (w * h) - intersect
                iou = intersect / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            # localize box in detector_mask and matching true_boxes
            if best_iou > 0:
                x_coord = np.floor(x).astype('int')
                y_coord = np.floor(y).astype('int')
                detector_mask[y_coord, x_coord, best_anchor] = 1
                yolo_box = np.array([y, x, h, w, box[4]])
                matching_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
    return matching_true_boxes, detector_mask, true_boxes_grid

def ground_truth_generator(dataset):
    '''
    Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.

    Parameters
    ----------
    - YOLO dataset. Generate batch:
        batch : tupple(images, annotations)
        batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        
    Returns
    -------
    - imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
    - detector_mask : tensor, shape (batch, size, GRID_H, GRID_W, anchors_count, 1)
        1 if bounding box detected by grid cell, else 0
    - matching_true_boxes : tensor, shape (batch_size, GRID_H, GRID_W, anchors_count, 5)
        Contains adjusted coords of bounding box in YOLO format
    - class_one_hot : tensor, shape (batch_size, GRID_H, GRID_W, anchors_count, class_count)
        One hot representation of bounding box label
    - true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
        true_boxes format : x, y, w, h, c, coords unit : grid cell
    '''
    for batch in dataset:
        # imgs
        imgs = batch[0]
        
        # true boxes
        true_boxes = batch[1]
        
        # matching_true_boxes and detector_mask
        batch_matching_true_boxes = []
        batch_detector_mask = []
        batch_true_boxes_grid = []
        
        for i in range(true_boxes.shape[0]):
            one_matching_true_boxes, one_detector_mask, true_boxes_grid = process_true_boxes(true_boxes[i],
                                                                                           ANCHORS,
                                                                                           IMAGE_W,
                                                                                           IMAGE_H)
            batch_matching_true_boxes.append(one_matching_true_boxes)
            batch_detector_mask.append(one_detector_mask)
            batch_true_boxes_grid.append(true_boxes_grid)
                
        detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype='float32')
        matching_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype='float32')
        true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), dtype='float32')
        
        # class one_hot
        matching_classes = tf.keras.backend.cast(matching_true_boxes[..., 4], 'int32')
        class_one_hot = tf.keras.backend.one_hot(matching_classes, CLASS + 1)[:,:,:,:,1:]
        class_one_hot = tf.cast(class_one_hot, dtype='float32')
        
        batch = (imgs, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid)
        yield batch
