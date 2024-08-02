import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.model_selection import KFold
from sklearn.utils import compute_class_weight

import wandb
from wandb.keras import WandbCallback


data_root = '/vol/biodata/retina/APTOS2019'
data_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
data_df['file_path'] = data_df['id_code'].apply(lambda x: os.path.join(data_root, 'pp_1024/train_images', x + '.png'))

k_folds = 5
batch_size = 4


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # Resize image with padding to 1024x1024
    # image = tf.image.resize(image, [1024, 1024])

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def augmentation_fn(image, label):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random rotation
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image, degree * np.pi / 180., interpolation='nearest')

    # Random brightness, saturation and contrast shifting
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_hue(image, 0.08)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    image = tf.image.random_contrast(image, 0.7, 1.3)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def load_image_train(image, label):
    image, label = parse_function(image, label)
    image, label = augmentation_fn(image, label)
    return image, label


kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
for idx, (train_idx, valid_idx) in enumerate(kf.split(data_df)):

    print(f"Training Fold {idx}... \n"
          f"TRAIN: {train_idx}, TEST: {valid_idx}")

    train_df, valid_df = data_df.loc[train_idx], data_df.loc[valid_idx]

    wandb.init(project='DeepSeeNet', entity="farrell236", group="x-val", name=f'fold_{idx}')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_df['file_path'], train_df['diagnosis']))
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.map(load_image_train, num_parallel_calls=8)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df['file_path'], valid_df['diagnosis']))
    valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=8)
    valid_dataset = valid_dataset.batch(batch_size)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=train_df['diagnosis'].unique(),
        y=train_df['diagnosis'])
    class_weights = dict(zip(train_df['diagnosis'].unique(), class_weights))

    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():

        model = tf.keras.Sequential([  # DeepSeeNet
            tf.keras.applications.inception_v3.InceptionV3(weights=None, include_top=False, input_shape=(1024, 1024, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu', name='global_dense1'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu', name='global_dense2'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(5, activation='softmax', name='global_predictions'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'CI/DeepSeeNet_{idx}.tf',
            monitor='val_accuracy', verbose=1, save_best_only=True
        )

        model.fit(
            train_dataset,
            validation_data=valid_dataset,
            class_weight=class_weights,
            steps_per_epoch=len(train_dataset),
            validation_steps=len(valid_dataset),
            epochs=100,
            callbacks=[checkpoint, WandbCallback()]
        )

    del model
    del train_dataset
    del valid_dataset

    wandb.join()
