# qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
# export QT_QPA_PLATFORM=offscreen

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# My libraries. Ugly hack to import from sister directory
import src.data_augmentation as data_augmentation
import src.segmentation as segmentation
import src.evaluation as evaluation
import src.util.utilities as util_
import src.util.flowlib as flowlib

class Uois:
    def __init__(self) -> None:

        dsn_config = {
            
            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10, 
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,
            
            # Misc
            'min_pixels_thresh' : 500,
            'tau' : 15.,
            
        }

        rrn_config = {
            
            # Sizes
            'feature_dim' : 64, # 32 would be normal
            'img_H' : 224,
            'img_W' : 224,
            
            # architecture parameters
            'use_coordconv' : False,
            
        }

        uois3d_config = {
            
            # Padding for RGB Refinement Network
            'padding_percentage' : 0.25,
            
            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,
            
            # Largest Connected Component for IMP module
            'use_largest_connected_component' : True,
            
        }

        checkpoint_dir = '/home/choiyj/uois/checkpoints/' # TODO: change this to directory of downloaded models
        dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                                dsn_filename,
                                                dsn_config,
                                                rrn_filename,
                                                rrn_config
                                                )
        
    def run(self, image_npy, seg_path):
        img = np.load(image_npy, allow_pickle=True, encoding='bytes').item()
        rgb = np.zeros((1, 480, 640, 3), dtype=np.float32)
        xyz = np.zeros((1, 480, 640, 3), dtype=np.float32)
        rgb[0] = data_augmentation.standardize_image(img['rgb'])
        xyz[0] = img['xyz']

        image = {
            'rgb' : data_augmentation.array_to_tensor(rgb),
            'xyz' : data_augmentation.array_to_tensor(xyz)
        }

        fg_masks, center_offsets, initial_masks, seg_masks = self.uois_net_3d.run_on_batch(image)

        seg_masks = seg_masks.cpu().numpy()
        fg_masks = fg_masks.cpu().numpy()
        center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
        initial_masks = initial_masks.cpu().numpy()

        num_objs = np.unique(seg_masks[0,...]).max() + 1

        seg_mask_plot = util_.get_color_mask(seg_masks[0,...], nc=num_objs)

        cv2.imwrite(seg_path, seg_mask_plot)

        return seg_mask_plot
    

def main():
    uois = Uois()
    example_images_dir = "/home/choiyj/catkin_ws/src/soomac/src/vision/a/dataset"
    imgs = sorted(glob.glob(example_images_dir + '/test_image_*.npy'))
    for i, img in enumerate(imgs):
        uois.run(img, 'segmask_'+str(i)+'.png')


if __name__ == "__main__":
    main()