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

    # Freeze the ViT layers to prevent Catastrophic Forgetting during initial fine-tuning
    vit_model.trainable = False

    # Define model architecture adding classification head
    # Widen the dense layer to 256 to avoid information bottleneck
    # Add Dropout for regularization
    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation=tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation='softmax')
    ], name='vision_transformer')

    # Compile the model with a lower initial learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model