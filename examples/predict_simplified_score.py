"""
Grades color fundus photographs using the AREDS Simplified Severity Scale.

usage: predict_simplified_score.py [-h] [-d DRUSEN] [-p PIGMENT]
                                   [-a ADVANCED_AMD] [-l LEFT_EYE]
                                   [-r RIGHT_EYE] [-g GPU] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DRUSEN, --drusen DRUSEN
                        Model file for Drusen
  -p PIGMENT, --pigment PIGMENT
                        Model file for Pigment
  -a ADVANCED_AMD, --advanced_amd ADVANCED_AMD
                        Model file for Advanced AMD
  -l LEFT_EYE, --left_eye LEFT_EYE
                        Image file for Left Eye
  -r RIGHT_EYE, --right_eye RIGHT_EYE
                        Image file for Right Eye
  -g GPU, --gpu GPU     Select GPU
  -v, --verbose         Increase output verbosity

"""
import argparse
import cv2

import matplotlib.pyplot as plt

from deepseenet import simplified_model


def load_and_plot_images(image_path1, image_path2, title):
    # Load the images using OpenCV
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert the images from BGR (OpenCV default) to RGB (matplotlib default)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Plot the images side by side
    plt.figure(figsize=(8, 4))

    # Display the first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Left Eye')
    plt.axis('off')

    # Display the second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('Right Eye')
    plt.axis('off')

    plt.tight_layout()

    # Set the main title
    plt.suptitle(title)

    # Show the plot
    plt.show()


if __name__ == '__main__':

    # Create the parser
    parser = argparse.ArgumentParser(description="Grades color fundus photographs using the AREDS Simplified Severity Scale")
    parser.add_argument('-d', '--drusen', type=str, help='Model file for Drusen')
    parser.add_argument('-p', '--pigment', type=str, help='Model file for Pigment')
    parser.add_argument('-a', '--advanced_amd', type=str, help='Model file for Advanced AMD')
    parser.add_argument('-l', '--left_eye', type=str, help='Image file for Left Eye')
    parser.add_argument('-r', '--right_eye', type=str, help='Image file for Right Eye')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Select GPU')
    parser.add_argument('-v', '--verbose', action='store_true', help='Increase output verbosity')
    args = parser.parse_args()

    ##### DEBUG OVERRIDE #####
    args.drusen = '../model_weights/drusen_model.h5'
    args.pigment = '../model_weights/pigment_model.h5'
    args.advanced_amd = '../model_weights/advanced_amd_model.h5'
    args.left_eye = ''
    args.right_eye = ''
    args.verbose = False

    clf = simplified_model.DeepSeeNetSimplifiedScore(args.drusen, args.pigment, args.advanced_amd)
    # score = clf.predict(args.left_eye, args.right_eye, verbose=args.verbose)
    # print(f'The simplified score: {score}')

    for pid in [10, 13, 15, 16, 17]:

        args.left_eye = f'../data/EyePACS/sample/{pid}_left.jpeg'
        args.right_eye = f'../data/EyePACS/sample/{pid}_right.jpeg'

        score, debug = clf.predict(args.left_eye, args.right_eye, verbose=args.verbose)
        print(f'The simplified score {pid}: {score} \t {debug}')
        load_and_plot_images(args.left_eye, args.right_eye, f'The simplified score {pid}: {score}')
