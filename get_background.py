import os
import cv2
import tqdm
import numpy as np
import matplotlib.pyplot as plt

ffhq_dir = "/mnt/lustre/wangzhibo/ffhq/images1024x1024"
ffhq_mask_dir = "/mnt/lustre/wangzhibo/ffhq/images_mask1024x1024"
output_dir = "/mnt/lustre/wangzhibo/ffhq/images_background512x512"


os.makedirs(output_dir, exist_ok=True)
for i in tqdm.tqdm(range(70000)):
    img = cv2.imread(os.path.join(ffhq_dir, '{:05d}.png').format(i), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (512, 512))
    img_mask = cv2.imread(os.path.join(ffhq_dir, '{:05d}.png').format(i), cv2.IMREAD_UNCHANGED)
    img_mask = ((img_mask[..., 0:1] == 255)*1.0) * ((img_mask[..., 1:2] == 255)*1.0) * ((img_mask[..., 2:3] == 255)*1.0)
    img *= (1-img_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '{:05d}.png'.format(i)))

