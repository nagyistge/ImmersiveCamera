# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:17:42 2016

@author: psy3061
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 18:08:10 2016

@author: psy3061
"""


import glob

import stitch
import seam as seamfunc
import numpy as np
import cv2
import findTransformMatrix as findTM
import cylindrical_projection as cyproj

import imutils
from timeit import default_timer as timer



print sorted(glob.glob("./input/*.jpg"))

images = list()



for img in sorted(glob.glob("./input/*.jpg")):
    images.append((imutils.resize(cv2.imread(img), width=1000)))
    #images.append(cv2.imread(img))
    
stitch_img = images[0]

h_matrix_list = list()
min_max_list = list()


for index in range(len(images)):
    if index != 0:
        M =  stitch.get_sift_homography(stitch_img,images[index])
        
        
        x_min, x_max, y_min, y_max = stitch.get_stitched_image_points(images[index],stitch_img,M)
        h_matrix_list.append(M)
        min_max_list.append([x_min, x_max, y_min, y_max])
        print x_min, x_max, y_min, y_max
        stitch_img = stitch.get_stitched_image(images[index],stitch_img,M)


cv2.imwrite("stitch.jpg",stitch_img)


# Get Total Width & Total Height From Last Min Max Values
total_width = min_max_list[-1][1] - min_max_list[-1][0]
total_height = min_max_list[-1][3] - min_max_list[-1][2]


warp_matrix_list = list()
for i in range(len(h_matrix_list)):
    transform_dist = [-min_max_list[i][0],-min_max_list[i][2]]    
    transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0,0,1]])
    warp_matrix_list.append(transform_array.dot(h_matrix_list[i]))




transform_matrix_list = list()
transformed_width_list = list()
transformed_height_list = list()

for i in range(len(images)):
    if i < len(images)-1:
        for j in range(i,len(warp_matrix_list)):
            if i == j:
                result_img = cv2.warpPerspective(images[i], warp_matrix_list[j], (total_width, total_height))
            else:
                result_img = cv2.warpPerspective(result_img, warp_matrix_list[j], (total_width, total_height))
        transform_matrix_list.append(findTM.findTransformMatrix(images[i], result_img))
        for x in range(images[i].shape[1],total_width):
            if sum(result_img[:,x]) == 0:
                transformed_width_list.append(x)
                break
        for y in range(images[i].shape[0],total_height):
            if sum(result_img[y,:]) == 0:
                transformed_height_list.append(y)
                break
    else:
        transform_matrix_list.append(np.diag([1,1,1]).astype(float))
        transformed_width_list.append(images[i].shape[1])
        transformed_height_list.append(images[i].shape[0])




trans_images = list()
for i in range(len(images)):
    if i == 0:
        trans_images.append(cv2.warpPerspective(images[i], transform_matrix_list[i], (transformed_width_list[i], total_height)))
    else:
        
        M = np.float32([[1,0,0],[0,1,-min_max_list[i-1][2]]])
    
        trans_images.append(cv2.warpAffine(cv2.warpPerspective(images[i], transform_matrix_list[i], (transformed_width_list[i], total_height)),M,(transformed_width_list[i], total_height)))




seam_list = list()

for i in range(len(images)-1):
    if i == 0:
        fromX = -min_max_list[i][0]
        if len(images) == 2:
            toX = images[i].shape[1]-50
        else:
            toX = -min_max_list[i+1][0]-50
    else:
        fromX = -min_max_list[i][0]+min_max_list[i-1][0]
        if i+1 < len(images)-1:
            toX = -min_max_list[i+1][0]+min_max_list[i-1][0]
        else:
            toX = images[i].shape[1] - min_max_list[i-1][0]+min_max_list[i-1][0]
    
    print fromX, toX

    imgSeam = trans_images[i]

    img_gray = cv2.cvtColor(imgSeam[:,fromX+20:toX-20,:], cv2.COLOR_RGB2GRAY)
    img_en = seamfunc.getEnergyImage(img_gray)
    
    seam_list.append(seamfunc.extractSeam(img_en)+fromX+20)



def findMaskLeftEnd(mask):
    for x in range(mask.shape[1]):
        if np.prod(mask[:,x]) != 0:
            return x
            break


basic_mask_list = list()


def createBasicMask(mask_num):
    i = mask_num
    mask_layer = np.fromfunction(lambda y,x : seam_list[i][y] >= x, trans_images[i].shape[:2], dtype=int)
    mask_layer = mask_layer.astype(int)
    mask = np.dstack((mask_layer,mask_layer,mask_layer))
    
    return mask,1-mask

# Create Basic Mask and Reverse of Basic Mask
for i in range(len(trans_images)):
    if i < len(trans_images)-1:
        basic_mask_list.append(createBasicMask(i))
    else:
        mask_ones = np.dstack((np.ones(trans_images[i].shape[:2]).astype(int),np.ones(trans_images[i].shape[:2]).astype(int),np.ones(trans_images[i].shape[:2]).astype(int)))
        basic_mask_list.append((mask_ones,1-mask_ones))

        
        

# Create Combined Mask for Each Image
combined_mask_list = list()
for i in range(len(trans_images)):
    if i == 0:
        combined_mask_list.append(basic_mask_list[i][0])
    else:        
        mask_before_reverse = basic_mask_list[i-1][1]           
        mask_current = basic_mask_list[i][0]
        if i == 1:
            combined_mask_list.append(np.hstack((mask_before_reverse[:,-min_max_list[i-1][0]:findMaskLeftEnd(mask_before_reverse)],mask_current[:,mask_before_reverse[:,-min_max_list[i-1][0]:findMaskLeftEnd(mask_before_reverse)].shape[1]:])))
        else:            
            combined_mask_list.append(np.hstack((mask_before_reverse[:,-min_max_list[i-1][0]+min_max_list[i-2][0]:findMaskLeftEnd(mask_before_reverse)],mask_current[:,mask_before_reverse[:,-min_max_list[i-1][0]+min_max_list[i-2][0]:findMaskLeftEnd(mask_before_reverse)].shape[1]:])))



def applyMaskToImage(num):
    return trans_images[num][:,:,:]*combined_mask_list[num]
 





# Apply Mask and Save it to Array

after_mask_trans_image_array = np.zeros((len(trans_images),total_height,total_width,3))


for i in range(len(trans_images)):
    if i == 0:
        after_mask_trans_image_array[i,:,0:transformed_width_list[0],:] = applyMaskToImage(i)
        
    else:
        after_mask_trans_image_array[i,:,-min_max_list[i-1][0]:transformed_width_list[i]-min_max_list[i-1][0],:] = applyMaskToImage(i)


final_combined_image = np.sum(after_mask_trans_image_array, 0)


cv2.imwrite("combine6.jpg",final_combined_image)





# Find Tilt Position

line_img = cv2.warpPerspective(np.ones((trans_images[0].shape[0],1,3)) * 255, transform_matrix_list[0], (transformed_width_list[0], total_height))

tilt_x = 0


for x in range(line_img.shape[1]):    
    if sum(line_img[:,x,:]) == 0:
        tilt_x = x
        break
    
    
pt2 = np.float32([[0,0],[total_width,0],[0,total_height],[total_width,total_height]])  
#point = np.float32([[0,0],[total_width,0],[tilt_x,total_height],[total_width,total_height]])
point = np.float32([[tilt_x,0],[total_width,170],[0,total_height-150],[total_width,total_height]])

## Afine Transformation for Straightening
straight_Transform_Matrix = cv2.getPerspectiveTransform( point, pt2 )  
straighted_img = cv2.warpPerspective( final_combined_image, straight_Transform_Matrix, (total_width, total_height ))  


cv2.imwrite("trans_combine3.jpg",straighted_img)



## Find Overlap Point of Left-End and Right-End
imageA = straighted_img[:,0:trans_images[0].shape[1],:]
imageB = straighted_img[:,-min_max_list[len(min_max_list)-1][0]:,:]


from_px = 0
to_px = 0

min_width = min(imageA.shape[1],imageB.shape[1])

compare_value_list = list()

while True:
    if from_px == 0 and to_px < min_width:
        to_px = to_px + 1
    else:
        from_px = from_px + 1
    
    if from_px == min_width:
        break

    #print from_px, to_px

    crop_imageA = imageA[:,from_px:to_px]
    crop_imageB = imageB[:,min_width-to_px:min_width-from_px]
    
    
    
    err = np.sum((crop_imageA.astype('float')-crop_imageB.astype('float'))**2)
    err /= float(crop_imageA.shape[0] * crop_imageA.shape[1])    
    
    compare_value_list.append(err)    

min_err_px = compare_value_list.index(min(compare_value_list))

#fig = plt.figure()
#plt.plot(compare_value_list)

start_point_calibrated_img = straighted_img[:,min_err_px:,:]

cv2.imwrite("start_point_calibrate3.jpg",start_point_calibrated_img)



start = timer()


# Apply Mask and Save it to Array

after_mask_trans_image_array = np.zeros((len(trans_images),total_height,total_width,3))

end = timer()

print (end-start)
start = timer()

for i in range(len(trans_images)):
    if i == 0:
        #after_mask_trans_image_array[i,:,0:transformed_width_list[0],:] = applyMaskToImage(i)
        
        after_mask_trans_image_array[i,:,0:transformed_width_list[0],:] = trans_images[i][:,:,:]*combined_mask_list[i]
        #after_mask_trans_image_array[i,:,0:transformed_width_list[0],:] = map(lambda img, mask: img * mask, trans_images[i][:,:,:],combined_mask_list[i])
    else:
        #after_mask_trans_image_array[i,:,-min_max_list[i-1][0]:transformed_width_list[i]-min_max_list[i-1][0],:] = applyMaskToImage(i)
        after_mask_trans_image_array[i,:,-min_max_list[i-1][0]:transformed_width_list[i]-min_max_list[i-1][0],:] = trans_images[i][:,:,:]*combined_mask_list[i]

end = timer()

print (end-start)

start = timer()


final_combined_image = np.sum(after_mask_trans_image_array, 0)


end = timer()

print (end-start)


start = timer()
straighted_img = cv2.warpPerspective( final_combined_image, straight_Transform_Matrix, (total_width, total_height ))  



start_point_calibrated_img = straighted_img[:,min_err_px:,:]

end = timer()

print (end-start)