from constants import *
from visualize import *
from data import *
import numpy as np
from scipy.spatial.distance import cdist


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

    r = np.random.uniform(0, np.sum(choices))
    upto = 0
    for c, w in enumerate(choices):
        if upto + w >= r:
            return c
        upto += w
    return 0

class KMeans:

    def __init__(self, k):

        self.k = k
        self.diff_thresh = 1
        self.distf = IoU_dist
        #self.distf = lambda x,y: (x[0]-y[0])**2 + (x[1]-y[1])**2

    def fit(self, data):
        initial_centroids = self.init_centroids_kpp(data)

        self.centroids, self.clusters = self.cluster_data(data, initial_centroids)
        return self.centroids, self.clusters
    
    def init_centroids_kpp(self, data):

        centroids = []
        
        random_index = np.random.randint(len(data))
        centroids.append(data[random_index])

        while len(centroids) < self.k:
            
            prob_array = np.apply_along_axis(lambda x:
                self.mindist2(x, centroids), 1, data)

            norm = sum(prob_array)
            prob_array /= (norm + 1e-8)
            
            new_index = weighted_choice(prob_array)
            centroids.append(data[new_index])

        return np.array(centroids)


    def mindist2(self, x, centroids):
        dists = np.apply_along_axis(lambda c: self.distf(x, c),1, centroids)
        return np.min(dists) * np.min(dists)


    def cluster_data(self, data, initial_centroids):
        centroids = initial_centroids
        clusters = []
        counter = 0
        while True:
            old_clusters = clusters 
            old_centroids = centroids

            clusters = self.clusterfy(data, centroids)

            centroids = self.recalc_centroids(data, clusters)

            # Kmeans stopping condition based on some centroid shift delta?
            if len(old_clusters)>0:
                num_diffs = np.sum(old_clusters != clusters)
                print("Iteration = %d, Delta = %d"%(counter, num_diffs), flush=True)
                
                if num_diffs <= self.diff_thresh:
                    break
            counter += 1

        return centroids, clusters

    def clusterfy(self, data, centroids):
        return np.apply_along_axis(lambda d:
            np.argmin(cdist([d], centroids, self.distf)[0]), 1, data)


    def recalc_centroids(self, data, clusters):
        
        new_centroids = []

        for centroid_index in range(self.k):        
            
            centroid_data_idxs = np.where(clusters==centroid_index)[0]
            centroid_data = data[centroid_data_idxs]
            new_centroids.append( np.mean(centroid_data, axis=0) )

        return np.array(new_centroids)

def compute_anchors(imgs_name, bbox, max_annot):
    wh = get_wh(imgs_name, bbox, max_annot)
    clustering = KMeans(BOX)
    centroids, _ = clustering.fit(wh)
    anchors = list(centroids.flatten())
    return anchors
