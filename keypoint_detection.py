import cv2
import numpy as np
from math import sqrt
from math import pi
from math import e
import math

image = cv2.imread('E:/CVIP/Project/Task_1/data/task2.jpg',0)
cv2.imshow('image1',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def flipKernel(kernel):

     #flip_kernel = np.asarray([[0.0 for i in range(kernel.shape[1])]for j in range(kernel.shape[0])])
     flip_kernel = kernel.copy()
     for x in range(kernel.shape[0]):
         for y in range(kernel.shape[1]):
             flip_kernel[x][y]=kernel[kernel.shape[0]-1-x][kernel.shape[1]-1-y]

     return flip_kernel


def convolution(image,kernel):
    #imageMat = np.asarray(imageMat)
    k_h,k_w  = kernel.shape
   
    # calculting image height and width
    i_h,i_w = image.shape

    # zero paddding to not to miss out edges
    x = k_h-1
    y = k_w-1

    m = (k_h-1)//2
    #n = (kernel.shape[1]-1)//2

    image_padding  = np.asarray([[0.0 for i in range(i_w+y)] for j in range(i_h+x)])
    image_padding[m:-m,m:-m] = image
    output = np.asarray([[0.0 for i in range(i_w+y)] for j in range(i_h+x)])
    
    kernel = flipKernel(kernel)

    for x in range(m,i_h+m):
        for y in range(m,i_w+m):
             for u in range(-m,m+1):
                 for v in range(-m,m+1):
                     output[x,y]+= kernel[u+m,v+m]*image_padding[u+x,v+y]

    return output[m:-m,m:-m]
    

def resize(img):
    
    height,width = img.shape

    resize = [[0 for i in range(width//2)] for i in range(height//2)]

    for i in range(height//2):
       # iterate through rows of image
       for j in range(width//2):
           resize[i][j] = img[i*2][j*2];

    return np.asarray(resize, dtype="uint8")

resize_image_octave2 = resize(image)
resize_image_octave3 = resize(resize_image_octave2)
resize_image_octave4 = resize(resize_image_octave3)

# gaussian kernel
gaussian_operator = np.asarray([[0.0 for i in range(7)]for j in range(7)])

# set of sigmas
octave_1_sigma = [1/sqrt(2),1,sqrt(2),2,2*sqrt(2)]
octave_2_sigma = [sqrt(2),2,2*sqrt(2),4,4*sqrt(2)]
octave_3_sigma = [2*sqrt(2),4,4*sqrt(2),8,8*sqrt(2)]
octave_4_sigma = [4*sqrt(2),8,8*sqrt(2),16,16*sqrt(2)]

# list of blur images obtained using gaussian operator
blur_image_set_octave1 = []
blur_image_set_octave2 = []
blur_image_set_octave3 = []
blur_image_set_octave4 = []

def getBlurImages(image,gaussian_operator,sigma):
     blur_images = []    
     for p in range(len(sigma)):
         sigma_operator = sigma[p]
   #print(sigma_operator)
         for i in range(gaussian_operator.shape[0]):
             x = i-gaussian_operator.shape[0]//2
             for j in range(gaussian_operator.shape[1]):
                 y = j-gaussian_operator.shape[1]//2
                 gaussian_operator[i,j] = (1/(2*math.pi*(sigma_operator**2)))*(math.e**-((x**2 + y**2)/(2*(sigma_operator**2))))
         blur_image = convolution(image,gaussian_operator)
         blur_images.append(blur_image)
         cv2.imshow("Blur Image - ",np.array(blur_image, dtype = 'uint8'))
         cv2.waitKey(0)
         cv2.destroyAllWindows()

     return blur_images

blur_image_set_octave1 = getBlurImages(image,gaussian_operator,octave_1_sigma)
print("blur_image_set_octave1- "+str(len(blur_image_set_octave1)))
blur_image_set_octave2 = getBlurImages(resize_image_octave2,gaussian_operator,octave_2_sigma)
blur_image_set_octave3 = getBlurImages(resize_image_octave3,gaussian_operator,octave_3_sigma)
blur_image_set_octave4 = getBlurImages(resize_image_octave4,gaussian_operator,octave_4_sigma)


# list of DoGs for each octave
gaussian_differce_octave1=[]
gaussian_differnnce_octave2=[]
gaussian_differnce_octave3=[]
gaussian_differnce_octave4=[]

# finging DoGs
def getDoGs(blur_image_set):
     gaussian_differnces = []
     for i in range(len(blur_image_set)-1):
         blur_image_1 = blur_image_set[i]
         blur_image_2 = blur_image_set[i+1]
         #DoG_1 = np.asarray([[0.0 for i in range(blur_image_1.shape[1])]for j in range(blur_image_1.shape[0])])
         DoG_1 = blur_image_1.copy()
         for x in range(blur_image_1.shape[0]):
             for y in range(blur_image_1.shape[1]):
         	     DoG_1[x][y] = blur_image_1[x][y] - blur_image_2[x][y]
         gaussian_differnces.append(DoG_1)
     return gaussian_differnces

gaussian_differnce_octave1 = getDoGs(blur_image_set_octave1)
print(len(gaussian_differnce_octave1))
gaussian_differnce_octave2 = getDoGs(blur_image_set_octave2)
gaussian_differnce_octave3 = getDoGs(blur_image_set_octave3)
gaussian_differnce_octave4 = getDoGs(blur_image_set_octave4)

# finding maxima and minima of GoDs
DoG_Octave2_1 = gaussian_differnce_octave2[0]
cv2.imshow("DoG_Octave1_1 - ",np.asarray(DoG_Octave2_1, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

DoG_Octave2_2 = gaussian_differnce_octave2[1]
cv2.imshow("DoG_Octave1_2 - ",np.asarray(DoG_Octave2_2, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


DoG_Octave2_3 = gaussian_differnce_octave2[2]
cv2.imshow("DoG_Octave1_3 - ",np.asarray(DoG_Octave2_3, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

DoG_Octave2_4 = gaussian_differnce_octave2[3]
cv2.imshow("DoG_Octave1_4 - ",np.asarray(DoG_Octave2_4, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# finding maxima and minima of GoDs
DoG_Octave3_1 = gaussian_differnce_octave3[0]
cv2.imshow("DoG_Octave1_1 - ",np.asarray(DoG_Octave3_1, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

DoG_Octave3_2 = gaussian_differnce_octave3[1]
cv2.imshow("DoG_Octave1_2 - ",np.asarray(DoG_Octave3_2, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()


DoG_Octave3_3 = gaussian_differnce_octave3[2]
cv2.imshow("DoG_Octave1_3 - ",np.asarray(DoG_Octave3_3, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

DoG_Octave3_4 = gaussian_differnce_octave3[3]
cv2.imshow("DoG_Octave1_4 - ",np.asarray(DoG_Octave3_4, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()



# calling findKeypoint()
# calculate min max of DoG
def findKeypoint(keypoint_detect,DoG_Octave1_1,DoG_Octave1_2,DoG_Octave1_3):
    for x in range(1,DoG_Octave1_2.shape[0]-1):
        for y in range(1,DoG_Octave1_2.shape[1]-1):
            isMax = False
            isMin = False
            # finding is Max
            val = DoG_Octave1_2[x][y]
            
            min_val = min(DoG_Octave1_1[x-1][y-1],DoG_Octave1_3[x-1][y-1],DoG_Octave1_2[x-1][y-1],DoG_Octave1_1[x-1][y],DoG_Octave1_3[x-1][y],DoG_Octave1_2[x-1][y],DoG_Octave1_1[x-1][y+1],
                          DoG_Octave1_3[x-1][y+1],DoG_Octave1_2[x-1][y+1],DoG_Octave1_1[x][y-1],DoG_Octave1_3[x][y-1],DoG_Octave1_2[x][y-1],
                          DoG_Octave1_1[x][y],DoG_Octave1_3[x][y],DoG_Octave1_1[x][y+1],DoG_Octave1_3[x][y+1],DoG_Octave1_2[x][y+1],
                          DoG_Octave1_1[x+1][y-1],DoG_Octave1_3[x+1][y-1],DoG_Octave1_2[x+1][y-1],DoG_Octave1_1[x+1][y],DoG_Octave1_3[x+1][y],
                          DoG_Octave1_2[x+1][y],DoG_Octave1_1[x+1][y+1],DoG_Octave1_3[x+1][y+1],DoG_Octave1_2[x+1][y+1])
            
            max_val = max(DoG_Octave1_1[x-1][y-1],DoG_Octave1_3[x-1][y-1],DoG_Octave1_2[x-1][y-1],DoG_Octave1_1[x-1][y],DoG_Octave1_3[x-1][y],DoG_Octave1_2[x-1][y],DoG_Octave1_1[x-1][y+1],
                          DoG_Octave1_3[x-1][y+1],DoG_Octave1_2[x-1][y+1],DoG_Octave1_1[x][y-1],DoG_Octave1_3[x][y-1],DoG_Octave1_2[x][y-1],
                          DoG_Octave1_1[x][y],DoG_Octave1_3[x][y],DoG_Octave1_1[x][y+1],DoG_Octave1_3[x][y+1],DoG_Octave1_2[x][y+1],
                          DoG_Octave1_1[x+1][y-1],DoG_Octave1_3[x+1][y-1],DoG_Octave1_2[x+1][y-1],DoG_Octave1_1[x+1][y],DoG_Octave1_3[x+1][y],
                          DoG_Octave1_2[x+1][y],DoG_Octave1_1[x+1][y+1],DoG_Octave1_3[x+1][y+1],DoG_Octave1_2[x+1][y+1])

            
            if(val>max_val or val<min_val):
                keypoint_detect[x][y] = 255

    return keypoint_detect    

# merging keypoints
def keypointMerge(keypoint1,keypoint2):
    keypoint_merge = np.asarray([[0.0 for i in range(keypoint1.shape[1])]for j in range(keypoint1.shape[0])])
    for i in range(keypoint1.shape[0]):
        for j in range(keypoint1.shape[1]):
            if(keypoint1[i][j] == 255 or keypoint2[i][j] == 255):
                  keypoint_merge[i][j]=255
    return keypoint_merge


def copyImage(image):
    copy = np.asarray([[0.0 for i in range(image.shape[1])]for j in range(image.shape[0])])
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            copy[x][y] = image[x][y]

#octave 1 keypoint 
keypoint_detect_1_octave1 = np.asarray([[0 for i in range(gaussian_differnce_octave1[0].shape[1])] for i in range(gaussian_differnce_octave1[0].shape[0])])
keypoint_detect_2_octave1 = np.asarray([[0 for i in range(gaussian_differnce_octave1[0].shape[1])] for i in range(gaussian_differnce_octave1[0].shape[0])])

keypoint_detect_1_octave1 = findKeypoint(keypoint_detect_1_octave1,gaussian_differnce_octave1[0],gaussian_differnce_octave1[1],gaussian_differnce_octave1[2])
cv2.imshow("Ocatve 1 Keypoint Image - 1",np.asarray(keypoint_detect_1_octave1, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_detect_2_octave1 = findKeypoint(keypoint_detect_2_octave1,gaussian_differnce_octave1[1],gaussian_differnce_octave1[2],gaussian_differnce_octave1[3])
cv2.imshow("Octave 1 Keypoint Image - 2 ",np.asarray(keypoint_detect_2_octave1, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_octave1_merge = keypointMerge(keypoint_detect_1_octave1,keypoint_detect_2_octave1)
cv2.imshow("Octave 1 Keypoint merge ",np.asarray(keypoint_octave1_merge, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = image.copy()
for i in range(keypoint_octave1_merge.shape[0]):
    for j in range(keypoint_octave1_merge.shape[1]):
        if(keypoint_octave1_merge[i][j] == 255):
            img1[i][j] = 255

cv2.imshow("Octave 1 Keypoint overlay image ",np.asarray(img1, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# declaring keypoint images for octave 2
keypoint_detect_1_octave2 = np.asarray([[0 for i in range(gaussian_differnce_octave2[0].shape[1])] for i in range(gaussian_differnce_octave2[0].shape[0])])
keypoint_detect_2_octave2 = np.asarray([[0 for i in range(gaussian_differnce_octave2[0].shape[1])] for i in range(gaussian_differnce_octave2[0].shape[0])])

keypoint_detect_1_octave2 = findKeypoint(keypoint_detect_1_octave2,gaussian_differnce_octave2[0],gaussian_differnce_octave2[1],gaussian_differnce_octave2[2])
cv2.imshow("Ocatve 1 Keypoint Image - 1",np.asarray(keypoint_detect_1_octave2, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_detect_2_octave2 = findKeypoint(keypoint_detect_2_octave2,gaussian_differnce_octave2[1],gaussian_differnce_octave2[2],gaussian_differnce_octave2[3])
cv2.imshow("Octave 1 Keypoint Image - 2 ",np.asarray(keypoint_detect_2_octave2, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_octave2_merge = keypointMerge(keypoint_detect_1_octave2,keypoint_detect_2_octave2)
cv2.imshow("Octave 1 Keypoint merge ",np.asarray(keypoint_octave2_merge, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = image.copy()
for i in range(keypoint_octave2_merge.shape[0]):
    for j in range(keypoint_octave2_merge.shape[1]):
        if(keypoint_octave2_merge[i][j] == 255):
            img2[i*2][j*2] = 255

cv2.imshow("Octave 1 Keypoint overlay image ",np.asarray(img2, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# declaring keypoint images for octave 3
keypoint_detect_1_octave3 = np.asarray([[0 for i in range(gaussian_differnce_octave3[0].shape[1])] for i in range(gaussian_differnce_octave3[0].shape[0])])
keypoint_detect_2_octave3 = np.asarray([[0 for i in range(gaussian_differnce_octave3[0].shape[1])] for i in range(gaussian_differnce_octave3[0].shape[0])])

keypoint_detect_1_octave3 = findKeypoint(keypoint_detect_1_octave3,gaussian_differnce_octave3[0],gaussian_differnce_octave3[1],gaussian_differnce_octave3[2])
cv2.imshow("Ocatve 1 Keypoint Image - 1",np.asarray(keypoint_detect_1_octave3, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_detect_2_octave3 = findKeypoint(keypoint_detect_2_octave3,gaussian_differnce_octave3[1],gaussian_differnce_octave3[2],gaussian_differnce_octave3[3])
cv2.imshow("Octave 1 Keypoint Image - 2 ",np.asarray(keypoint_detect_2_octave3, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_octave3_merge = keypointMerge(keypoint_detect_1_octave3,keypoint_detect_2_octave3)
cv2.imshow("Octave 1 Keypoint merge ",np.asarray(keypoint_octave3_merge, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

img3 = image.copy()
for i in range(keypoint_octave3_merge.shape[0]):
    for j in range(keypoint_octave3_merge.shape[1]):
        if(keypoint_octave3_merge[i][j] == 255):
            img3[i*4][j*4] = 255

cv2.imshow("Octave 1 Keypoint overlay image ",np.asarray(img3, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

# declaring keypoint images for octave 4
keypoint_detect_1_octave4 = np.asarray([[0 for i in range(gaussian_differnce_octave4[0].shape[1])] for i in range(gaussian_differnce_octave4[0].shape[0])])
keypoint_detect_2_octave4 = np.asarray([[0 for i in range(gaussian_differnce_octave4[0].shape[1])] for i in range(gaussian_differnce_octave4[0].shape[0])])

keypoint_detect_1_octave4 = findKeypoint(keypoint_detect_1_octave4,gaussian_differnce_octave4[0],gaussian_differnce_octave4[1],gaussian_differnce_octave4[2])
cv2.imshow("Ocatve 1 Keypoint Image - 1",np.asarray(keypoint_detect_1_octave4, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_detect_2_octave4 = findKeypoint(keypoint_detect_2_octave4,gaussian_differnce_octave4[1],gaussian_differnce_octave4[2],gaussian_differnce_octave4[3])
cv2.imshow("Octave 1 Keypoint Image - 2 ",np.asarray(keypoint_detect_2_octave4, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoint_octave4_merge = keypointMerge(keypoint_detect_1_octave4,keypoint_detect_2_octave4)
cv2.imshow("Octave 1 Keypoint merge ",np.asarray(keypoint_octave4_merge, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()

img4 = image.copy()
for i in range(keypoint_octave4_merge.shape[0]):
    for j in range(keypoint_octave4_merge.shape[1]):
        if(keypoint_octave4_merge[i][j] == 255):
            img4[i*8][j*8] = 255

cv2.imshow("Octave 1 Keypoint overlay image ",np.asarray(img4, dtype = 'uint8'))
cv2.waitKey(0)
cv2.destroyAllWindows()





