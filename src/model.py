import os
import sys
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import src.config as config

# In Python 3.12, vit-keras might not be fully compatible due to tensorflow-addons.
# We mock tensorflow-addons to map to built-in keras functionalities.
class MockTFALayers:
    GELU = tf.keras.layers.Activation('gelu')
    @staticmethod
    def GELU(*args, **kwargs):
        return tf.keras.layers.Activation('gelu')

    @staticmethod
    def AdamW(*args, **kwargs):
        # Fallback to standard Adam as AdamW may not be readily available in legacy Keras
        return tf.keras.optimizers.Adam(*args, **kwargs)

class MockTFA:
    pass

mock_tfa = MockTFA()
mock_tfa.layers = MockTFALayers()
mock_tfa.optimizers = MockTFALayers()

sys.modules['tensorflow_addons'] = mock_tfa
sys.modules['tensorflow_addons.layers'] = mock_tfa.layers
sys.modules['tensorflow_addons.optimizers'] = mock_tfa.optimizers

from vit_keras import vit

def build_vit_model():
    """
    Builds and compiles the Vision Transformer model.
    """
    # Initialize ViT base model
    vit_model = vit.vit_b32(
        image_size=config.IMAGE_SIZE,
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=config.NUM_CLASSES
    )

    # Freeze the ViT layers initially for Two-Phase Fine-Tuning
    vit_model.trainable = False

    # Define model architecture adding classification head
    # Addressing the architectural bottleneck by expanding the Dense layer
    # and introducing Dropout for regularization.
    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.gelu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')
    ], name='vision_transformer')

    # Compile the model with the initial head learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_HEAD)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model