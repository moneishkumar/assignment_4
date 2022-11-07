import pdb
import argparse
from re import template
from tkinter import image_names
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.signal import correlate2d
from scipy.interpolate import interp2d
import glob
import pdb


def main():
    pass

if __name__=="__main__":

    folder = "./../../data/output/"
    img_list = glob.glob(folder+'*.png')
    img_list.sort()

    img = cv2.imread(img_list[0], 0)
    search_space_co = cv2.selectROI(img)
    #search_space_co = (115, 300, 186, 228)

    template_co = cv2.selectROI(img)
    #template_co = (162, 399, 74, 53)
    template_ = img[template_co[1]:template_co[1]+template_co[3], \
                    template_co[0]:template_co[0]+template_co[2]]

    template_mean = np.mean(template_)
    box = np.ones((template_co[2], template_co[3]))/ (template_co[3]*template_co[2])

    template_sub = template_ - template_mean
    template_var = np.var(template_sub)

    shifts = []
    focus_image = np.zeros((img.shape[0], img.shape[1], 3))
    
    t = np.arange(img.shape[1])
    s = np.arange(img.shape[0])
    x_old = 0
    y_old = 0

    for idxs, img_name in enumerate(img_list):
        print(idxs)
        print(img_name)
        img = cv2.imread(img_name, 0)
        img_search = img[search_space_co[1]:search_space_co[1]+search_space_co[3],\
                         search_space_co[0]:search_space_co[0]+search_space_co[2]]

        img_dash = correlate2d(img_search, box, mode='same')
        img_sub = img_search-img_dash
        img_var = np.var(img_sub)

        h = correlate2d(img_sub, template_sub, mode='same')
        h = h/(np.sqrt(img_var*template_var))
        
        h_w = (((h - np.min(h))/np.max(h))*255).astype(np.uint8)
        cv2.imwrite(f'./../../output/h_{idxs}.png', h_w)
        
        idx = np.argmax(h)
        y = idx//search_space_co[2]
        x = idx%search_space_co[2]
        
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(img_sub)

        # plt.subplot(1,3,2)
        # plt.imshow(template_sub)
        
        # plt.subplot(1,3,3)
        # plt.imshow(img)
        # plt.scatter(search_space_co[0] + x, search_space_co[1] + y)

        # plt.show()
        # pdb.set_trace()

        img_c = cv2.imread(img_name)
        for channel in range(3):
            
            f = interp2d(t,s,img_c[:,:,channel])
            if idxs !=0:
                diff_x = x_1 - x
                diff_y = y_1 - y
                
                t_new = t - diff_x
                s_new = s - diff_y  
            else:
                t_new = t
                s_new = s
                y_1 = y
                x_1 = x
            
            #print(x,y, x_old, y_old, np.max(t_new - t), np.min(s_new - s), idxs)
            #pdb.set_trace()
            focus_image[:,:,channel] += f(t_new,s_new)

        
        shifts.append([x,y])
        pass
    focus_image  = focus_image/len(img_list)
    plt.imshow(focus_image.astype(np.uint8))
    plt.show()