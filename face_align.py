import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
import sys

detector = MTCNN()

person = "me"
if len(sys.argv) > 1:
    person = sys.argv[1]
    

image = cv2.imread(person + ".jpg")
rows,cols,colors = image.shape
print(rows)
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
keypoints = result[0]['keypoints']  

s = (keypoints['right_eye'][1] - keypoints['left_eye'][1])/(keypoints['right_eye'][0] - keypoints['left_eye'][0])

deg = np.degrees(np.arctan(s))

M = cv2.getRotationMatrix2D((cols/2,rows/2),deg,1)

'''
cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
'''

out = cv2.warpAffine(image,M,(cols,rows))

cv2.imshow('ww', out)
cv2.waitKey(0)

result = detector.detect_faces(out)

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
keypoints = result[0]['keypoints']  
re = keypoints['right_eye']
le = keypoints['left_eye']
curr_pix = re[0] - le[0]
s = 50.0 / float(curr_pix) 
print(s)
m,n,_ = out.shape

    
out = cv2.resize(out, (int(s * n), int(s * m)))
m,n,_ = out.shape
result = detector.detect_faces(out)
keypoints = result[0]['keypoints']  
print(out.shape)

re = keypoints['right_eye']
le = keypoints['left_eye']
print(re)
print(le)
x = int((re[0] + le[0])/2)
print(x)
y = int((re[1] + le[1])/2) 

out = out[y - 112:y + (250 - 112), x - 125:x + 125, :]
print(out.shape)

cv2.imshow('ww', out)
cv2.waitKey(0)

cv2.imwrite(person + "_aligned.jpg", out)


