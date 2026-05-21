import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit
import src.config as config

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

    # Define model architecture adding classification head
    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(11, activation=tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')
    ], name='vision_transformer')

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
