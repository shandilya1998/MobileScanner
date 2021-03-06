from src.object_detection.constants import *
import matplotlib.pyplot as plt
import numpy as np

def visualize_classes(imgs_name, bbox, max_annot):
    count = np.zeros(len(LABELS)+1)
    def f(label):
        count[int(label)] += 1
        return label
    
    for i in range(imgs_name.shape[0]):
        for j in range(max_annot):
            f(bbox[i][j][4])
    f, ax = plt.subplots(1, figsize = (10, 10))
    ax.bar(LABELS, count[1:])
    ax.set_ylabel('Count')
    ax.set_xlabel('Labels')
    ax.set_title('Label distribution in all images')
    f.savefig('label_distribution.png')
    return count[1:]

def visualize_box_size(wh):
    f, ax = plt.subplots(1, figsize = (10, 10))
    plt.scatter(wh[:,0],wh[:,1],alpha=0.1)
    ax.set_title("Box Size Clusters")
    ax.set_xlabel("normalized width")
    ax.set_ylabel("normalized height")
    f.savefig('box_size_clusters.png')
