import numpy as np
import cv2
from tqdm import tqdm

bases = None
with open('bases.npy', 'rb') as f:
    bases = np.load(f)

print(bases.shape)

me = cv2.imread('me_aligned.jpg', cv2.IMREAD_GRAYSCALE).astype(float)/255.0

coeffs = []
for b in tqdm(bases):
    coeffs.append((me * b).sum() / (b*b).sum())

with open('coeffs.npy', 'wb') as f:
    np.save(f, np.array(coeffs))

