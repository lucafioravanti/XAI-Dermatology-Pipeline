import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import src.config as config

def step_decay(epoch):
    """Learning rate decay schedule."""
    # Start from a much lower LR to match the new compilation
    initial_lrate = 1e-4
    drop = 0.5 # Gentler drop
    epochs_drop = 5.0 # Drop more frequently
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def get_callbacks(model_save_path='vit_model.h5'):
    """Returns a list of callbacks for training."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5, # Slightly higher patience since learning rate is lower
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        model_save_path,
        save_best_only=True
    )

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)

    return [early_stopping, model_checkpoint, lr_scheduler]

def train_model(model, train_ds, val_ds):
    """
    Executes the training loop using tf.data.Dataset.
    """
    callbacks = get_callbacks()

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )

    return history
