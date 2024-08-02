import logging

import numpy as np
import tensorflow as tf

from deepseenet.utils import preprocess_image

def load_model(model_file):
    logging.info('Loading the model: %s', model_file)
    return tf.keras.models.load_model(model_file, compile=False)


def get_simplified_score(scores):
    """
    Get AREDS simplified severity score from drusen size, pigmentary abnormality, and advanced AMD.

    Args:
        scores: a dict of individual risk factors

    Returns:
        a score of 0-5
    """
    def has_adv_amd(score):
        return True if score == 1 else False

    def has_pigment(score):
        return True if score == 1 else False

    def has_large_drusen(score):
        return True if score == 2 else False

    def has_intermediate_drusen(score):
        return True if score == 1 else False

    score = 0
    if has_adv_amd(scores['advanced_amd'][0]):
        score += 5
    if has_adv_amd(scores['advanced_amd'][1]):
        score += 5
    if has_pigment(scores['pigment'][0]):
        score += 1
    if has_pigment(scores['pigment'][1]):
        score += 1
    if has_large_drusen(scores['drusen'][0]):
        score += 1
    if has_large_drusen(scores['drusen'][1]):
        score += 1
    if has_intermediate_drusen(scores['drusen'][0]) \
            and has_intermediate_drusen(scores['drusen'][1]):
        score += 1

    return 5 if score >= 5 else score


class DeepSeeNetSimplifiedScore(object):
    def __init__(self, drusen_model, pigment_model, advanced_amd_model):
        """
        Args:
            drusen_model: Path or file object.
            pigment_model: Path or file object.
            advanced_amd_model: Path or file object.
        """
        self.models = {
            'drusen': load_model(drusen_model),
            'pigment': load_model(pigment_model),
            'advanced_amd': load_model(advanced_amd_model),
        }

    def predict(self, x_left, x_right, verbose=False):
        """
        Generates simplified severity score for one left eye and one right eye

        Args:
            x_left: input data of the left eye, as a Path or file object.
            x_right: input data of the right eye, as a Path or file object.
            verbose: Verbosity mode (bool).

        Returns:
            Numpy array of scores of 0-5
        """
        # assert x_left.shape[0] == x_right.shape[0]
        scores = {}
        for model_name, model in self.models.items():
            left_logits = model.predict(preprocess_image(x_left), verbose=verbose)
            right_logits = model.predict(preprocess_image(x_right), verbose=verbose)
            left_score = np.argmax(left_logits, axis=1)[0]
            right_score = np.argmax(right_logits, axis=1)[0]
            scores[model_name] = (left_score, right_score)
            scores[f'{model_name}_logits'] = (left_logits, right_logits)
        if verbose:
            logging.info('Risk factors: %s', scores)

        return get_simplified_score(scores), scores
