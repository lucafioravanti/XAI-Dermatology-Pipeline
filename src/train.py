import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import src.config as config

def step_decay(epoch):
    """Learning rate decay schedule."""
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def get_callbacks(model_save_path='vit_model.h5'):
    """Returns a list of callbacks for training."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        model_save_path,
        save_best_only=True
    )

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)

    return [early_stopping, model_checkpoint, lr_scheduler]

def train_model(model, train_data, val_data):
    """
    Executes the training loop.
    """
    train_images, train_labels = train_data
    val_images, val_labels = val_data

    callbacks = get_callbacks()

    print("Starting training...")
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )

    return history
