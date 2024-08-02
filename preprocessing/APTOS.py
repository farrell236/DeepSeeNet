import cv2
import os
import tqdm

import numpy as np
import pandas as pd

from utils import _pad_to_square, _get_retina_bb, rgb_clahe

# import matplotlib.pyplot as plt


data_root = '/vol/biodata/retina/APTOS2019'

data_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
# data_df = pd.read_csv(os.path.join(data_root, 'test.csv'))


for idx, row in tqdm.tqdm(data_df.iterrows(), total=len(data_df)):

    # Load Image
    image_file = os.path.join(data_root, 'train_images', row['id_code']+'.png')
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image); plt.show()

    # Localise and center retina image
    x, y, w, h, mask = _get_retina_bb(image)
    image = image[y:y + h, x:x + w, :]
    image = _pad_to_square(image, border=0)
    image = cv2.resize(image, (1024, 1024))

    # Center retina mask
    mask = np.uint8(mask[..., None] > 0)
    mask = mask[y:y + h, x:x + w, :]
    mask = _pad_to_square(mask, border=0)
    mask = cv2.resize(mask, (1024, 1024), 0, 0, cv2.INTER_NEAREST)

    # Apply CLAHE pre-processing
    image = rgb_clahe(image)

    # Display or save image
    # plt.imshow(image); plt.show()
    # plt.imshow(mask); plt.show()
    cv2.imwrite(os.path.join(data_root, 'pp_1024/train_images', row['id_code']+'.png'), image)
    cv2.imwrite(os.path.join(data_root, 'pp_1024/train_mask', row['id_code']+'.png'), mask)
