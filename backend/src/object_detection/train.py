# -*- coding: utf-8 -*-
import os
import glob
import re
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import imgaug as ia
from datetime import datetime
from imgaug import augmenters as iaa
from src.object_detection.yolov2 import *
from src.object_detection.mobilenetv2 import *
from src.object_detection.yolo_lite import *
from src.object_detection.tiny_yolo import *
from src.object_detection.loss import *
from src.object_detection.gnd_truth import *

import tensorflow as tf

def run(
    learning_rate,
    beta1,
    beta2,
    eps,
    job_dir,
    seed,
    log_interval,
    train_batch_size,
    val_batch_size,
    architecture, 
    plot_model,
):

    tf.random.set_seed(
        seed
    )


    TRAIN_BATCH_SIZE = train_batch_size
    VAL_BATCH_SIZE = val_batch_size
    print('Tensorflow version : {}'.format(tf.__version__))
    print('GPU : {}'.format(tf.config.list_physical_devices('GPU')))
    
    model = None
    if architecture == 'yolov2':
        model = get_yolov2_model(plot_model)
    elif architecture == 'yolo-lite':
        model = get_yolo_lite(plot_model)
    elif architecture == 'tiny-yolo':
        model = get_tiny_yolo(plot_model)
    elif architecture == 'mobilenet':
        model = get_mobilenet_model(plot_model)
    else:
        raise ValueError('Expected one of yolov2, yolo-lite, tiny-yolo or mobilenet as architecture value, but got '+architecture)
    
    """# 1. Data generator"""

    def parse_annotation(ann_dir, img_dir, labels):
        '''
        Parse XML files in PASCAL VOC format.
        
        Parameters
        ----------
        - ann_dir : annotations files directory
        - img_dir : images files directory
        - labels : labels list
        
        Returns
        -------
        - imgs_name : numpy array of images files path (shape : images count, 1)
        - true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
            annotation format : xmin, ymin, xmax, ymax, class
            xmin, ymin, xmax, ymax : image unit (pixel)
            class = label index
        '''
     
        max_annot = 0
        imgs_name = []
        annots = []
        
        # Parse file
        for ann in sorted(os.listdir(ann_dir)):
            annot_count = 0
            boxes = []
            tree = ET.parse(ann_dir + ann)
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    imgs_name.append(os.path.join(img_dir, elem.text))
                if 'width' in elem.tag:
                    w = int(elem.text)
                if 'height' in elem.tag:
                    h = int(elem.text)
                if 'object' in elem.tag or 'part' in elem.tag:
                    box = np.zeros((5))
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            box[4] = labels.index(attr.text) + 1 # 0:label for no bounding box
                        if 'bndbox' in attr.tag:
                            annot_count += 1
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    box[0] = math.floor(float(dim.text)*IMAGE_W/1024)
                                if 'ymin' in dim.tag:
                                    box[1] = math.floor(float(dim.text)*IMAGE_H/1024)
                                if 'xmax' in dim.tag:
                                    box[2] = math.ceil(float(dim.text)*IMAGE_W/1024)
                                if 'ymax' in dim.tag:
                                    box[3] = math.ceil(float(dim.text)*IMAGE_H/1024)
                    boxes.append(np.asarray(box))
            
            """
            if w != IMAGE_W or h != IMAGE_H :
                print('Image size error')
                break
            """
     
            annots.append(np.asarray(boxes))
            

            if annot_count > max_annot:
                max_annot = annot_count
               
        # Rectify annotations boxes : len -> max_annot
        imgs_name = np.array(imgs_name)
        true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
        for idx, boxes in enumerate(annots):
            true_boxes[idx, :boxes.shape[0], :5] = boxes
            
        return imgs_name, true_boxes

    """## 1.1. Dataset"""

    def parse_function(img_obj, true_boxes):
        x_img_string = tf.io.read_file(img_obj)
        x_img = tf.image.decode_png(x_img_string, channels=NUM_C) # dtype=tf.uint8
        x_img = tf.image.convert_image_dtype(x_img, tf.float32) # pixel value /255, dtype=tf.float32, channels : RGB
        x_img = tf.image.resize(x_img, (IMAGE_H, IMAGE_W))
        return x_img, true_boxes

    def get_dataset(img_dir, ann_dir, labels, batch_size):
        '''
        Create a YOLO dataset
        
        Parameters
        ----------
        - ann_dir : annotations files directory
        - img_dir : images files directory
        - labels : labels list
        - batch_size : int
        
        Returns
        -------
        - YOLO dataset : generate batch
            batch : tupple(images, annotations)
            batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        Note : image pixel values = pixels value / 255. channels : RGB
        '''
        imgs_name, bbox = parse_annotation(ann_dir, img_dir, LABELS)
        dataset = tf.data.Dataset.from_tensor_slices((imgs_name, bbox))
        dataset = dataset.shuffle(len(imgs_name))
        dataset = dataset.repeat()
        dataset = dataset.map(parse_function, num_parallel_calls=6)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        print('-------------------')
        print('Dataset:')
        print('Images count: {}'.format(len(imgs_name)))
        print('Step per epoch: {}'.format(len(imgs_name) // batch_size))
        print('Images per epoch: {}'.format(batch_size * (len(imgs_name) // batch_size)))
        return dataset

    train_dataset = None
    train_dataset= get_dataset(train_image_folder, train_annot_folder, LABELS, TRAIN_BATCH_SIZE)

    val_dataset = None
    val_dataset= get_dataset(val_image_folder, val_annot_folder, LABELS, VAL_BATCH_SIZE)

    # Test dataset

    def test_dataset(dataset):
        for batch in dataset:
            img = batch[0][0]
            label = batch[1][0]
            plt.figure(figsize=(2,2))
            f, (ax1) = plt.subplots(1,1, figsize=(10, 10))
            ax1.imshow(img)
            ax1.set_title('Input image. Shape : {}'.format(img.shape))
            for i in range(label.shape[0]):
                box = label[i,:]
                box = box.numpy()
                x = box[0]
                y = box[1]
                w = box[2] - box[0]
                h = box[3] - box[1]
                if box[4] == 1:
                    color = (0, 1, 0)
                else:
                    color = (1, 0, 0)
                rect = patches.Rectangle((x, y), w, h, linewidth = 2, edgecolor=color,facecolor='none')
                ax1.add_patch(rect)
            break
            
    test_dataset(train_dataset)

    """## 1.2. Data augmentation"""

    def augmentation_generator(dataset):
        '''
        Augmented batch generator from a dataset

        Parameters
        ----------
        - YOLO dataset
        
        Returns
        -------
        - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            batch : tupple(images, annotations)
            batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        '''
        for batch in dataset:
            # conversion tensor->numpy
            img = batch[0].numpy()
            boxes = batch[1]. numpy()
            # conversion bbox numpy->ia object
            ia_boxes = []
            for i in range(img.shape[0]):
                ia_bbs = [ia.BoundingBox(x1=bb[0],
                                           y1=bb[1],
                                           x2=bb[2],
                                           y2=bb[3]) for bb in boxes[i]
                          if (bb[0] + bb[1] +bb[2] + bb[3] > 0)]
                ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(IMAGE_W, IMAGE_H)))
            # data augmentation
            seq = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.Multiply((0.4, 1.6)), # change brightness
                #iaa.ContrastNormalization((0.5, 1.5)),
                #iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
                ])
            #seq = iaa.Sequential([])
            seq_det = seq.to_deterministic()
            img_aug = seq_det.augment_images(img)
            img_aug = np.clip(img_aug, 0, 1)
            boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
            # conversion ia object -> bbox numpy
            for i in range(img.shape[0]):
                boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
                for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                    boxes[i,j,0] = bb.x1
                    boxes[i,j,1] = bb.y1
                    boxes[i,j,2] = bb.x2
                    boxes[i,j,3] = bb.y2
            # conversion numpy->tensor
            batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
            #batch = (img_aug, boxes)
            yield batch

    aug_train_dataset = augmentation_generator(train_dataset)

    test_dataset(aug_train_dataset)

    # Ground true generator

    train_gen = ground_truth_generator(aug_train_dataset)
    val_gen = ground_truth_generator(val_dataset)

    # batch
    img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(train_gen)

    # y
    matching_true_boxes = matching_true_boxes[0,...]
    detector_mask = detector_mask[0,...]
    class_one_hot = class_one_hot[0,...]
    y = tf.keras.backend.concatenate((matching_true_boxes[...,0:4], detector_mask, class_one_hot), axis = -1)

    # y_hat
    y_hat = model.predict_on_batch(img)[0,...]

    # img
    img = img[0,...]

    # display prediction (Yolo Confidence value)
    plt.figure(figsize=(2,2))
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 10))
    ax1.imshow(img)
    ax1.set_title('Image')

    ax2.matshow((tf.keras.backend.sum(y[:,:,:,4], axis=2))) # YOLO Confidence value
    ax2.set_title('Ground truth')
    ax2.xaxis.set_ticks_position('bottom')

    ax3.matshow(tf.keras.backend.sum(y_hat[:,:,:,4], axis=2)) # YOLO Confidence value
    ax3.set_title('Prediction')
    ax3.xaxis.set_ticks_position('bottom')

    f.tight_layout()

    """# 2. Train"""


    # test loss

    # get batch
    img, detector_mask, matching_true_boxes, class_one_hot, true_boxe_grid = next(train_gen)

    # first image in batch
    img = img[0:1]
    detector_mask = detector_mask[0:1]
    matching_true_boxes = matching_true_boxes[0:1]
    class_one_hot = class_one_hot[0:1]
    true_boxe_grid = true_boxe_grid[0:1]

    # predict
    y_pred = model.predict_on_batch(img)

    # plot img, ground truth and prediction
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 5))
    ax1.imshow(img[0,...])
    ax1.set_title('Image')
    ax2.matshow(tf.keras.backend.sum(detector_mask[0,:,:,:,0], axis=2)) # YOLO Confidence value
    ax2.set_title('Ground truth, count : {}'.format(tf.keras.backend.sum(tf.cast(detector_mask  > 0., tf.int32))))
    ax2.xaxis.set_ticks_position('bottom')
    ax3.matshow(tf.keras.backend.sum(y_pred[0,:,:,:,4], axis=2)) # YOLO Confidence value
    ax3.set_title('Prediction')
    ax3.xaxis.set_ticks_position('bottom')
    f.tight_layout()

    # loss info
    loss, sub_loss = LOSS(detector_mask, matching_true_boxes, class_one_hot, true_boxe_grid, y_pred, info = True)

    """## 2.1. Training"""

    # gradients
    def grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes, training=True):
        with tf.GradientTape() as tape:
            y_pred = model(img, training)
            loss, sub_loss = LOSS(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)
        return loss, sub_loss, tape.gradient(loss, model.trainable_variables)

    # save weights
    def save_best_weights(model, name, val_loss_avg):
        # delete existing weights file
        files = glob.glob(os.path.join(job_dir, 'checkpoint/', name + '*'))
        for file in files:
            os.remove(file)
        # create new weights file
        name = name + '_' + str(val_loss_avg) + '.h5'
        path_name = os.path.join(job_dir, 'checkpoint/', name)
        model.save_weights(path_name)
    # log (tensorboard)
    def log_loss(loss, val_loss, step):
        tf.summary.scalar('loss', loss, step)
        tf.summary.scalar('val_loss', val_loss, step)

    # training
    def train(epochs, model, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name = 'train'):
        '''
        Train YOLO model for n epochs.
        Eval loss on training and validation dataset.
        Log training loss and validation loss for tensorboard.
        Save best weights during training (according to validation loss).

        Parameters
        ----------
        - epochs : integer, number of epochs to train the model.
        - model : YOLO model.
        - train_dataset : YOLO ground truth and image generator from training dataset.
        - val_dataset : YOLO ground truth and image generator from validation dataset.
        - steps_per_epoch_train : integer, number of batch to complete one epoch for train_dataset.
        - steps_per_epoch_val : integer, number of batch to complete one epoch for val_dataset.
        - train_name : string, training name used to log loss and save weights.
        
        Notes :
        - train_dataset and val_dataset generate YOLO ground truth tensors : detector_mask,
          matching_true_boxes, class_one_hot, true_boxes_grid. Shape of these tensors (batch size, tensor shape).
        - steps per epoch = number of images in dataset // batch size of dataset
        
        Returns
        -------
        - loss history : [train_loss_history, val_loss_history] : list of average loss for each epoch.
        '''
        num_epochs = epochs
        steps_per_epoch_train = steps_per_epoch_train
        steps_per_epoch_val = steps_per_epoch_val
        train_loss_history = []
        val_loss_history = []
        best_val_loss = 1e6
        
        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=eps)
        
        # log (tensorboard)
        summary_writer = tf.summary.create_file_writer(os.path.join(job_dir, 'logs/', train_name), flush_millis=20000)
        summary_writer.set_as_default()
       
        # training
        for epoch in range(num_epochs):
            epoch_loss = []
            epoch_val_loss = []
            epoch_val_sub_loss = []
            print('Epoch {} :'.format(epoch))
            # train
            for batch_idx in range(steps_per_epoch_train):
                img, detector_mask, matching_true_boxes, class_one_hot, true_boxes =  next(train_dataset)
                loss, _, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss.append(loss)
                print('-', end='')
            print(' | ', end='')
            # val
            for batch_idx in range(steps_per_epoch_val):
                img, detector_mask, matching_true_boxes, class_one_hot, true_boxes =  next(val_dataset)
                loss, sub_loss, grads = grad(model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes, training=False)
                epoch_val_loss.append(loss)
                epoch_val_sub_loss.append(sub_loss)
                print('-', end='')

            loss_avg = np.mean(np.array(epoch_loss))
            val_loss_avg = np.mean(np.array(epoch_val_loss))
            sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
            train_loss_history.append(loss_avg)
            val_loss_history.append(val_loss_avg)
            
            # log
            log_loss(loss_avg, val_loss_avg, epoch)
            
            # save
            if val_loss_avg < best_val_loss:
                save_best_weights(model, train_name, val_loss_avg)
                best_val_loss = val_loss_avg
            
            print(' loss = {:.4f}, val_loss = {:.4f} (conf={:.4f}, class={:.4f}, coords={:.4f})'.format(
                loss_avg, val_loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))
            
        return [train_loss_history, val_loss_history]

    results = train(EPOCHS, model, train_gen, val_gen, 10, 2, 'training_1')

    plt.plot(results[0])
    plt.plot(results[1])
    """# 3. Results"""

    def display(file, model, score_threshold, iou_threshold):
        '''
        Display predictions from YOLO model.

        Parameters
        ----------
        - file : string list : list of images path.
        - model : YOLO model.
        - score_threshold : threshold used for filtering predicted bounding boxes.
        - iou_threshold : threshold used for non max suppression.
        '''
        # load image
        image = cv2.imread(file)

        input_image = image[:,:,::-1]
        input_image = image / 255.
        input_image = np.expand_dims(input_image, 0)

        # prediction
        y_pred = model.predict_on_batch(input_image)

        # post prediction process
        # grid coords tensor
        coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
        coord_y = tf.transpose(coord_x, (0,2,1,3,4))
        coords = tf.tile(tf.concat([coord_x,coord_y], -1), [TRAIN_BATCH_SIZE, 1, 1, 5, 1])
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

        # Reshape
        pred_xy = pred_xy[0,...]
        pred_wh = pred_wh[0,...]
        box_conf = box_conf[0,...]
        box_class_prob = box_class_prob[0,...]

        # Convert box coords from x,y,w,h to x1,y1,x2,y2
        box_xy1 = pred_xy - 0.5 * pred_wh
        box_xy2 = pred_xy + 0.5 * pred_wh
        boxes = tf.keras.backend.concatenate((box_xy1, box_xy2), axis=-1)

        # Filter boxes
        box_scores = box_conf * box_class_prob
        box_classes = tf.keras.backend.argmax(box_scores, axis=-1) # best score index
        box_class_scores = tf.keras.backend.max(box_scores, axis=-1) # best score
        prediction_mask = box_class_scores >= score_threshold
        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        # Scale box to image shape
        boxes = boxes * IMAGE_H

        # Non Max Supression
        selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=iou_threshold)
        boxes = tf.keras.backend.gather(boxes, selected_idx)
        scores = tf.keras.backend.gather(scores, selected_idx)
        classes = tf.keras.backend.gather(classes, selected_idx)
        
        # Draw image
        plt.figure(figsize=(2,2))
        f, (ax1) = plt.subplots(1,1, figsize=(10, 10))
        ax1.imshow(image[:,:,::-1])
        count_detected = boxes.shape[0]
        ax1.set_title('Detected objects count : {}'.format(count_detected))
        for i in range(count_detected):
            box = boxes[i,...]
            x = box[0]
            y = box[1]
            w = box[2] - box[0]
            h = box[3] - box[1]
            classe = classes[i].numpy()
            if classe == 0:
                color = (0, 1, 0)
            else:
                color = (1, 0, 0)
            rect = patches.Rectangle((x.numpy(), y.numpy()), w.numpy(), h.numpy(), linewidth = 3, edgecolor=color,facecolor='none')
            ax1.add_patch(rect)
        f.savefig(os.path.join(job_dir, 'test', datetime.now().strftime('val_image_%Y%m%d_%H%M%S')))

    x_files =  glob.glob('data/val/image/*.png')

    score = SCORE_THRESHOLD
    iou_threshold = IOU_THRESHOLD

    score = 0.45
    iou_threshold = 0.3

    for file in x_files[::3]:
        display(file, model, score, iou_threshold)
