import logging

import numpy as np
import tensorflow as tf


def preprocess_image(image_path):
    logging.debug('Processing: %s', image_path)
    img = crop2square(tf.keras.utils.load_img(image_path)).resize((224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')
    return x


def crop2square(img):
    """
    Crop the image to a square based on the short edge.

    Args:
        img: PIL Image instance.

    Returns:
        A PIL Image instance.
    """
    short_side = min(img.size)
    x0 = (img.size[0] - short_side) / 2
    y0 = (img.size[1] - short_side) / 2
    x1 = img.size[0] - x0
    y1 = img.size[1] - y0
    return img.crop((x0, y0, x1, y1))
