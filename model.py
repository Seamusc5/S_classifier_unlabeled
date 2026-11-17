import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable()
class HorizontalMirror(layers.Layer):
    def call(self, inputs):
        return tf.reverse(inputs, axis=[2])


@tf.keras.utils.register_keras_serializable()
class ReflectionDifference(layers.Layer):
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] // 2
        return inputs[:batch_size] - inputs[batch_size:]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])


def build_chirality_cnn(input_shape=(35, 35, 1)):
    """Build a compact classifier whose mirrored outputs are exact complements."""
    inputs = layers.Input(shape=input_shape)
    mirrored = HorizontalMirror()(inputs)
    paired = layers.Concatenate(axis=0)([inputs, mirrored])

    features = layers.Conv2D(16, 3, padding="same", activation="relu")(paired)
    features = layers.MaxPooling2D(pool_size=2)(features)
    features = layers.Conv2D(24, 3, padding="same", activation="relu")(features)
    features = layers.MaxPooling2D(pool_size=2)(features)
    features = layers.Conv2D(32, 3, padding="same", activation="relu")(features)
    features = layers.GlobalAveragePooling2D()(features)
    features = layers.Dense(16, activation="relu")(features)
    scores = layers.Dense(1)(features)

    logits = ReflectionDifference()(scores)
    outputs = layers.Activation("sigmoid")(logits)
    return models.Model(inputs=inputs, outputs=outputs, name="ChiralCNN")
