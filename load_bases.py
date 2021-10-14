import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


bases = []
i = 0
stop = 5000
for root, dirs, files in os.walk("lfw-deepfunneled", topdown=True):
    for name in files:
        path = os.path.join(root, name)
        if path.endswith('.jpg'):
            base = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(float)/255.0
            bases.append(base)
        if i > stop:
            break
        i += 1
    if i > stop:
        break
            

print(i)
bases = np.array(bases)      
np.random.shuffle(bases)

fp = np.memmap('bases.dat', dtype='float32', mode='w+', shape=bases.shape)
fp[:] = bases[:]
fp.flush()

