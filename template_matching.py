import numpy as np

import cv2

template = cv2.imread('E:/CVIP/Project/Task_1/data/task3/new_template.jpg',0)

image = cv2.imread('E:/CVIP/Project/Task_1/data/task3/pos_15.jpg',0)

width,height = template.shape[::-1]

# height = template.shape[0]

# width = template.shape[1]

# new_image = cv2.CreateImage((width,height),IPL_DEPTH_8U,1)

# cv2.Zero(new_image)

image = cv2.GaussianBlur(image,(3,3),0)

image_lap = cv2.Laplacian(image,cv2.CV_32F)

template = cv2.Laplacian(template, cv2.CV_32F)

result = cv2.matchTemplate(image_lap,template,cv2.TM_CCOEFF)

result = cv2.normalize(result, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
 
cv2.imshow('image',result)

cv2.waitKey(0)

cv2.destroyAllWindows()

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

t_left = max_loc

print(min_loc)

print(max_val)

b_right = (t_left[0]+width ,t_left[1] + height)

cv2.rectangle(image,b_right,t_left,(0,0,255),2)

threshold = 1.0

#location = np.where( result >= threshold)

#for pt in zip(*location[::-1]):
    #cv2.rectangle(image, pt, (pt[0] + width, pt[1] + height), (0,0,255), 2)

cv2.imshow('image',image)

cv2.waitKey(0)

cv2.destroyAllWindows()
