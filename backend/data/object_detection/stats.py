import cv2
import numpy as np
import os
from tqdm import tqdm 
import matplotlib.pyplot as plt
count = {}
f = 'images'
fol = ['train', 'val']
"""
for folder in fol:
    avg = {}
    for path in tqdm(os.listdir(os.path.join(f, folder))):
        path = os.path.join(f, folder, path)
        im = cv2.imread(path)
        unique_elements, counts_elements = np.unique(im, return_counts=True)
        unique_elements = unique_elements.tolist() 
        counts_elements = counts_elements.tolist()
        for i in range(len(unique_elements)):
            if unique_elements[i] not in count.keys():
                count[unique_elements[i]] = counts_elements[i] 
            else:
                count[unique_elements[i]] += counts_elements[i]

print(count)"""
count = {'Black (0)': 31840770, 'White (255)': 2928289278}
f, ax = plt.subplots(1, 1, figsize = (5, 5))
ax.bar(count.keys(), count.values(), align = 'center', color='crimson')
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
ax.set_xlabel('pixel values')
ax.set_ylabel('value frequency')
ax.set_title('Pixel Value distribution')
f.savefig('pixel_stats.png')
plt.show()
