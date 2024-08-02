"""
Predict the drusen size of color fundus photographs.

usage: predict_drusen.py [-h] [-d DRUSEN] [-e EYE_IMAGE] [-g GPU] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DRUSEN, --drusen DRUSEN
                        Model file for Drusen
  -e EYE_IMAGE, --eye_image EYE_IMAGE
                        Image file for Eye
  -g GPU, --gpu GPU     Select GPU
  -v, --verbose         Increase output verbosity
"""
import argparse

import numpy as np
import tensorflow as tf

from deepseenet.utils import preprocess_image


drusen_size = {
    0: 'small/none',
    1: 'intermediate',
    2: 'large'
}

def get_drusen_size(score):
    y = np.argmax(score, axis=1)[0]
    return drusen_size[y]


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Predict the drusen size of color fundus photographs")
    parser.add_argument('-d', '--drusen', type=str, help='Model file for Drusen')
    parser.add_argument('-e', '--eye_image', type=str, help='Image file for Eye')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Select GPU')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    ##### DEBUG OVERRIDE #####
    args.drusen = '../model_weights/drusen_model.h5'
    args.eye_image = '../data/left_eye.jpg'
    args.verbose = False

    clf = tf.keras.models.load_model(args.drusen, compile=False)
    x = preprocess_image(args.eye_image)
    score = clf.predict(x, verbose=args.verbose)
    print('The drusen score:', score)
    print('The drusen size:', get_drusen_size(score))
