# qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
# export QT_QPA_PLATFORM=offscreen

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import json
from time import time
import glob

import open3d as o3d
import open3d.core as o3c

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

        return img['rgb'], seg_mask_plot


def extract_objects_from_image(rgb_image, segmask):
    # 고유 라벨 값을 추출 (배경을 제외한)
    unique_labels = np.unique(segmask)
    unique_labels = unique_labels[unique_labels != 0]  # 배경 라벨(0)을 제외

    cropped_images = []
    
    for label in unique_labels:
        # 현재 라벨에 해당하는 마스크 생성
        mask = np.where(segmask == label, 255, 0).astype(np.uint8)
        
        # 객체의 바운딩 박스 추출
        offset = 10
        x, y, w, h = cv2.boundingRect(mask)
        print(x, y, w, h)
        # RGB 이미지에서 객체 영역 크롭
        x0 = max(0, x-offset)
        y0 = max(0, y-offset)
        x1 = min(x+w+offset, 640)
        y1 = min(y+h+offset, 480)
        cropped_img = rgb_image[y0:y1, x0:x1]
        cropped_img = add_padding(cropped_img, [100, 100])
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_images.append((label, cropped_img))
        print(cropped_img.shape)

    return cropped_images

def add_padding(image, target_size):
    h, w, _ = image.shape
    target_h, target_w = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded_image = cv2.copyMakeBorder(
        resized_image, 
        pad_h, target_h - new_h - pad_h, 
        pad_w, target_w - new_w - pad_w, 
        cv2.BORDER_CONSTANT, 
        value=[0, 0, 0]
    )
    return padded_image


def main():
    uois = Uois()
    example_images_dir = "/home/choiyj/catkin_ws/src/soomac/src/vision/a"
    imgs = sorted(glob.glob(example_images_dir + '/test_image_*.npy'))
    for i, img in enumerate(imgs):
        rgb, seg = uois.run(img, 'test/segmask_'+str(i)+'.png')
        seg = cv2.imread(f'/home/choiyj/catkin_ws/src/soomac/src/vision/uois/test/segmask_{i}.png', cv2.IMREAD_GRAYSCALE)
        cropped_images = extract_objects_from_image(rgb, seg)

        for idx, img in enumerate(cropped_images):
            cv2.imwrite(f'/home/choiyj/catkin_ws/src/soomac/src/vision/a/test/cropped/cropped_object_{i}_{idx}.png', img[1])



if __name__ == "__main__":
    main()