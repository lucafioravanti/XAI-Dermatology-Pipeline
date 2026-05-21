import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import src.config as config

def get_callbacks(model_save_path='vit_model.h5', patience=3):
    """Returns a list of callbacks for training."""
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    model_checkpoint = ModelCheckpoint(
        model_save_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

    return [early_stopping, model_checkpoint, lr_scheduler]

def unfreeze_model(model):
    """Unfreezes the ViT base model and recompiles for fine-tuning."""
    # The ViT model is the first layer in our Sequential model
    vit_layer = model.layers[0]
    vit_layer.trainable = True

    # Recompile with the fine-tuning learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_FINETUNE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model unfrozen and recompiled for fine-tuning.")
    return model

def train_model(model, train_dataset, val_dataset):
    """
    Executes the Two-Phase training loop.
    """

    print("--- Phase 1: Training the Classification Head ---")
    # In Phase 1, the ViT is frozen (set in model.py)
    callbacks_head = get_callbacks(model_save_path='vit_model_head.h5', patience=3)

    history_head = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_HEAD,
        callbacks=callbacks_head
    )

    print("\n--- Phase 2: Fine-Tuning the Entire Model ---")
    model = unfreeze_model(model)

    callbacks_finetune = get_callbacks(model_save_path='vit_model_finetuned.h5', patience=5)

    history_finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_FINETUNE,
        callbacks=callbacks_finetune
    )

    return history_head, history_finetune
