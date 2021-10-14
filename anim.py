import numpy as np
import cv2
from tqdm import tqdm
import sys
from time import time

start_time = time()
bases = None
newfp = np.memmap('bases.dat', dtype='float32', mode='r', shape=(5001, 250, 250))
bases = newfp[:5000]
print('load time', time() - start_time)



person = "me"
if len(sys.argv) > 1:
    person = sys.argv[1]

me = cv2.imread(person +'_aligned.jpg', cv2.IMREAD_GRAYSCALE).astype(float)/255.0

A = []

for b in tqdm(bases):
    A.append(b[35:200,60:190].flatten())

A = np.array(A).T

if False:
    
    print(A.shape)
    start_time = time()
    u,s,v = np.linalg.svd(A)
    print('svd time', time() - start_time)
    np.savez('svd', u=u, s=s, v=v)

if True:
    svd = np.load('svd.npz')
    u = svd['u']
    s = svd['s']

    print(s.shape)
    u = u[:,:500]
    u = A
    x,_,_,_ = np.linalg.lstsq(u, me[35:200, 60:190].flatten())
    
    with open(person + '_weights.npy', 'wb') as f:
        np.save(f, x)

with open(person + '_weights.npy', 'rb') as f:
    x = np.load(f)
print(x)

tot_img = np.zeros((165,130))
for i in tqdm(range(len(u[0,:]))):
    base = u[:,i].reshape(165,130).astype(float)
    tot_img += x[i]*base
    cv2.imshow('res', cv2.hconcat([base / base.max(), tot_img + 0.3*base, me[35:200, 60:190]]))
    cv2.waitKey(10)


cv2.imshow('res', cv2.hconcat([tot_img, me[35:200, 60:190]]))
cv2.waitKey(0)


    
    


