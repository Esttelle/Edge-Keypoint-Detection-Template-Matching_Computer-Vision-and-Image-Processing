# importing open cv package to read and display
import cv2
import numpy as np

img = cv2.imread('E:/CVIP/Project/Task_1/data/task1.png',0)
cv2.imshow('image1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

def flipKernel(kernel):

     flip_kernel=kernel.copy()

     for x in range(kernel.shape[0]):
         for y in range(kernel.shape[1]):
             flip_kernel[x][y]=kernel[kernel.shape[0]-1-x][kernel.shape[1]-1-y]

     return flip_kernel



def convolution(image,kernel):
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    # calculting image height and width
    i_h = image.shape[0]
    i_w = image.shape[1]

    # zero paddding to not to miss out edges
    x = k_h-1
    y = k_w-1

    image_padding  = np.asarray([[0 for i in range(i_w+y)] for j in range(i_h+x)],dtype='float32')
    image_padding[1:-1,1:-1] = image
    #output = image.copy()
    output  = np.asarray([[0 for i in range(image.shape[1])] for j in range(image.shape[0])],dtype='float32')

    for x1 in range(image.shape[0]):
        for y1 in range(image.shape[1]):
            output[x1,y1]=(kernel*image_padding[x1:x1+k_w,y1:y1+k_h]).sum()  
    
    return output

# declaring kernel
Cx = [[0 for i in range(3)]for j in range(3)]
Cx[0][0] = -1
Cx[0][1] =  0
Cx[0][2] =  1
Cx[1][0] = -2
Cx[1][1] =  0
Cx[1][2] =  2
Cx[2][0] = -1
Cx[2][1] =  0
Cx[2][2] =  1

Cy = [[0 for i in range(3)]for j in range(3)]
Cy[0][0] = -1
Cy[0][1] = -2
Cy[0][2] = -1
Cy[1][0] =  0
Cy[1][1] =  0
Cy[1][2] =  0
Cy[2][0] =  1
Cy[2][1] =  2
Cy[2][2] =  1

Cx_array=flipKernel(np.asarray(Cx))
Cy_array=flipKernel(np.asarray(Cy))
output_x = convolution(img,Cx_array)
output_y = convolution(img,Cy_array)
cv2.imshow('Edge detection x direction',output_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Edge detection y direction',output_y)
cv2.waitKey(0)
cv2.destroyAllWindows()


# finding maximum of output array
max_val=255


# normalizing X-direction and Y-direction by dividing max value
Xresult = [[0.0 for i in range(output_x.shape[1])]for j in range(output_x.shape[0])]
result_x = np.asarray(Xresult,dtype='float32')
for i in range(output_x.shape[0]):
	for j in range(output_x.shape[1]):
		#if output_x[i][j]<0:
		result_x[i][j]=(output_x[i][j])/max_val

Yresult = [[0.0 for i in range(output_y.shape[1])]for j in range(output_y.shape[0])]
result_y = np.asarray(Yresult,dtype='float32')
for i in range(output_y.shape[0]):
    for j in range(output_y.shape[1]):
        #if output_x[i][j]<0:
        result_y[i][j]=(output_y[i][j])/max_val

print(result_x)
print("............")
cv2.imshow("result X- ",result_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result_y)
print("............")
cv2.imshow("result Y- ",result_y)
cv2.waitKey(0)
cv2.destroyAllWindows()




			