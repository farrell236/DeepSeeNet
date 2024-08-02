import tensorflow as tf


def RiskFactorModel(n_classes=2, input_shape=(224, 224, 3), weights='imagenet', name='RiskFactorModel'):
    return tf.keras.Sequential([
        tf.keras.applications.inception_v3.InceptionV3(weights=weights, include_top=False, input_shape=input_shape),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu', name='global_dense1'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', name='global_dense2'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(n_classes, activation='softmax', name='global_predictions'),
    ], name=name)


if __name__ == '__main__':

    model = RiskFactorModel(n_classes=2, input_shape=(224, 224, 3))
    print(model.summary())

    a=1
