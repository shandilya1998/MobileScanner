from src.object_detection.constants import *
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import imgaug as ia
from imgaug import augmenters as iaa
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def compute_iou(bb_1, bb_2):

    xa0, ya0, xa1, ya1 = bb_1
    xb0, yb0, xb1, yb1 = bb_2

    intersec = (min([xa1, xb1]) - max([xa0, xb0]))*(min([ya1, yb1]) - max([ya0, yb0]))

    union = (xa1 - xa0)*(ya1 - ya0) + (xb1 - xb0)*(yb1 - yb0) - intersec

    return intersec / union

def IoU_dist(x, c):
    return 1. - compute_iou([0,0,x[0],x[1]], [0,0,c[0],c[1]])

def get_wh(imgs_name, bbox, max_annot):
    wh = []
    for i in range(imgs_name.shape[0]):
        for j in range(max_annot):
            w = 0
            h = 0
            if bbox[i][j][0] == 0 and bbox[i][j][1] == 0 and bbox[i][j][2] == 0 and bbox[i][j][3] == 0:
                continue
            else:
                w = (bbox[i][j][1] - bbox[i][j][0])/IMAGE_W
                h = (bbox[i][j][3] - bbox[i][j][2])/IMAGE_H
            temp = [w,h]
            wh.append(temp)
    wh = np.array(wh)
    return wh

def weighted_choice(choices):
    r = np.random.uniform(0, np.sum(choices, -1))
    upto = 0
    for c, w in enumerate(choices):
        if upto + w >= r:
            return c
        upto += w
    return 0

def compute_anchors(imgs_name, bbox, max_annot):
    wh = get_wh(imgs_name, bbox, max_annot)
    kmeans = KMeans(n_clusters = BOX, random_state=0).fit(wh)
    centroids = kmeans.cluster_centers_
    anchors = list(centroids.flatten())
    return anchors

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
                                box[0] = math.floor(float(dim.text))
                            if 'ymin' in dim.tag:
                                box[1] = math.floor(float(dim.text))
                            if 'xmax' in dim.tag:
                                box[2] = math.ceil(float(dim.text))
                            if 'ymax' in dim.tag:
                                box[3] = math.ceil(float(dim.text))
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
        
    return imgs_name, true_boxes, max_annot

def parse_annotation_top(ann_dir, img_dir, labels):
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
                for child in elem:
                    if elem.find('name').text in labels:
                        box = np.zeros((5))
                        for attr in list(elem):
                            if 'name' in attr.tag:
                                box[4] = labels.index(attr.text) + 1 # 0:label for no bounding box
                            if 'bndbox' in attr.tag:
                                annot_count += 1
                                for dim in list(attr):
                                    if 'xmin' in dim.tag:
                                        box[0] = math.floor(float(dim.text))
                                    if 'ymin' in dim.tag:
                                        box[1] = math.floor(float(dim.text))
                                    if 'xmax' in dim.tag:
                                        box[2] = math.ceil(float(dim.text))
                                    if 'ymax' in dim.tag:
                                        box[3] = math.ceil(float(dim.text))
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
        
    return imgs_name, true_boxes, max_annot

"""## 1.1. Dataset"""

def parse_function(img_obj, true_boxes):
    x_img_string = tf.io.read_file(img_obj)
    x_img = tf.image.decode_png(x_img_string, channels=NUM_C) # dtype=tf.uint8
    x_img = tf.image.convert_image_dtype(x_img, tf.float32) # pixel value /255, dtype=tf.float32, channels : RGB
    x_img = tf.image.resize(x_img, (IMAGE_H, IMAGE_W))
    return x_img, true_boxes

def get_dataset(img_dir, ann_dir, labels, batch_size, compute = False, top = True):
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
    imgs_name = None
    bbox = None
    max_annot = None
    if top:
        imgs_name, bbox, max_annot = parse_annotation_top(ann_dir, img_dir, LABELS)
    else:
        imgs_name, bbox, max_annot = parse_annotation(ann_dir, img_dir, LABELS)
    visualize_classes(imgs_name, bbox, max_annot)
    if compute:
        global ANCHORS
        ANCHORS = compute_anchors(imgs_name, bbox, max_annot)
        print(ANCHORS)
        print(len(ANCHORS))
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
            #iaa.Multiply((0.4, 1.6)), # change brightness
            #iaa.ContrastNormalization((0.5, 1.5)),
            iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
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
