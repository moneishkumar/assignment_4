import pdb
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter


def main():
    pass

def generate_mosaic(light_field: np.ndarray) -> None:
    lenslet_size = light_field.shape[0]

    count = 1
    for i in range(lenslet_size):
        for j in range(lenslet_size):
            plt.subplot(lenslet_size, lenslet_size, count)
            plt.imshow(light_field[i,j,:,:,:])
            plt.axis('off')
            count +=1

    plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1  ,
                    wspace=0,
                    hspace=0)
    plt.savefig("./../../output/1_2.png", bbox_inches='tight', dpi=2000)
    #plt.show()


def create_lightfield(image: np.ndarray, lenslet_size: int) -> np.ndarray:
    h, w, c, = image.shape
    
    light_field = np.zeros((lenslet_size, lenslet_size, h//lenslet_size, w//lenslet_size, 3))

    for j in range(0,h, lenslet_size):
        for i in range(0,w, lenslet_size):  
            light_field[:,:,j//lenslet_size,i//lenslet_size,:] = image[j:j+lenslet_size,i:i+lenslet_size,:]

    return light_field

def integrate_lightfield(image: np.ndarray, d: list, mask_u, mask_v) -> np.ndarray:
    
    _, _, h, w, c = image.shape 
    
    
    u,v = get_uv()
    integrated_image = np.zeros((h, w, c, len(d)))
    f = {}
    s = (np.arange(h)) #y
    t = (np.arange(w)) #x

    denomionator = np.sum(mask_u)*np.sum(mask_v)

    for channel in range(3):
        for i in range(16):
            for j in range(16):
                if mask_v[j] == 1 and mask_u[i] == 1:
                    f = interp2d(t,s,image[i,j,:,:,channel])
                    for idx, d_val in enumerate(d):  
                        # change i,j  and a negative sign                             
                        integrated_image[:, :, channel, idx] += f(t + v[j]*d_val, s - u[i]*d_val)
                        #pdb.set_trace()

    integrated_image = (integrated_image/denomionator).astype(np.uint8)
    return integrated_image

def get_uv():

    lensletSize = 16;
    maxUV = (lensletSize - 1) / 2;
    u = np.arange(lensletSize) - maxUV;
    v = np.arange(lensletSize) - maxUV;

    return u,v 

def get_infocus_and_depth(focal_stack: np.ndarray, sigma_1, sigma_2, depths):
    
    h, w, c, d = focal_stack.shape
    
    I_l = np.zeros((h, w, d))
    I_lfreq = np.zeros((h, w, d))
    I_hfreq = np.zeros((h, w, d))
    w_s = np.zeros((h, w, d))

    for depth in range(d):
        I_l[:,:,depth] = cv2.cvtColor(focal_stack[:, :, :, depth], cv2.COLOR_RGB2XYZ)[:,:,1]
        I_lfreq[:,:,depth] = gaussian_filter(I_l[:,:,depth], sigma_1)
        I_hfreq[:,:,depth] = I_l[:,:,depth] - I_lfreq[:,:,depth] 
        w_s[:,:,depth] = gaussian_filter(I_hfreq[:,:,depth]**2, sigma_2)

    all_infocus_image = np.zeros((h,w,c))

    for channel in range(3):
        all_infocus_image[:,:,channel] = np.sum(focal_stack[:,:,channel,:]*w_s, axis = -1)/np.sum(w_s, axis = -1)

    depth_image = np.sum(w_s*np.array(depths), axis = -1)/np.sum(w_s, axis = -1)
    #pdb.set_trace()
    depth_image = depth_image - np.min(depth_image)
    depth_image = depth_image/ np.max(depth_image)

    return all_infocus_image.astype(np.uint8), (depth_image*255).astype(np.uint8)

def confocal_sterio(light_field, depths, apertures):
    
    AFI = []
    for aperture in apertures:

        mask_u = np.ones(16)
        mask_v = np.ones(16)
        zero = (16 - aperture)//2
        mask_u[:zero] = 0
        mask_u[-zero:] = 0

        mask_v[:zero] = 0
        mask_v[-zero:] = 0

        AFI.append(integrate_lightfield(light_field, depths, mask_u, mask_v))

    AFI = np.array(AFI)
    variance_afi = np.var(AFI, axis = 0)
    variance_sum = np.sum(variance_afi, axis = 2)
    idx = np.argmax(variance_sum, axis = -1)
    
    # for i in range(idx.shape[0]):
    #     for j in range(idx.shape[1]):
    #         idx[i,j] = depths[idx[i,j]]

    return (idx/np.max(idx))*255


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='CompPhoto HW4')
    parser.add_argument('--method',default='PiecewiseBilateral', help='Method that needs to be done')


    args = parser.parse_args()

    im = cv2.imread('./../../data/chessboard_lightfield.png')
    patch_size = 16
    
    depths = [-1.2, -1, -0.7, -0.5, -0.3, 0]
    apertures = [4, 6, 8, 10, 12, 14]

    #depths = [-0.05,-0.01,0,0.01,0.05]
    #depths = [5, 1, 0, 1, 5]
    h, w, c, = im.shape

    # 1.1
    light_field = create_lightfield(im, patch_size).astype(np.uint8)    
    
    # 1.2
    #generate_mosaic(light_field)


    #1.3
    integrated_image = integrate_lightfield(light_field, depths, np.ones(16), np.ones(16))

    for i in range(integrated_image.shape[3]):
        plt.imshow(integrated_image[:,:,:,i])
        plt.savefig(f'./../../output/intergrated_{i}.png')

    #1.4: All-in-focus image and depth from focus

        # COLOR_RGB2XYZ

    sigma_1 = 0.7
    sigma_2 = 5
    
    all_infocus, depth_image = get_infocus_and_depth(integrated_image, sigma_1, sigma_2, depths)

    cv2.imwrite(f'./../../output/all_infocus_{sigma_1}_{sigma_2}.png', all_infocus)
    cv2.imwrite(f'./../../output/depth_{sigma_1}_{sigma_2}.png', depth_image)

    #1.5 Focal-aperture stack and confocal stereo

    depth_map = confocal_sterio(light_field, depths, apertures)

    cv2.imwrite(f'./../../output/depth_confocal.png', depth_map)

    


    