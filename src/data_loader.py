import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import src.config as config

def load_image(image_path, label):
    """Loads, resizes, and normalizes an image using TensorFlow operations."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment_image(image, label):
    """Applies basic data augmentation suitable for medical images."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Note: Medical images can be safely rotated. tf.image.rot90 is simple and effective.
    # For more complex rotations, we could use tf.keras.layers.RandomRotation,
    # but applying it in tf.data pipeline with raw tf ops is sometimes faster.
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    return image, label

def create_tf_dataset(image_paths, labels, is_training=False):
    """Creates an efficient tf.data.Dataset from file paths and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle only for training
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(image_paths))

    # Map the image loading function
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data augmentation only for training
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def load_and_preprocess_data(apply_smote=True):
    """
    Loads metadata, creates image paths, splits data, optionally applies oversampling,
    and returns tf.data.Dataset objects for training, validation, and testing.
    """
    print("Loading metadata...")
    if not os.path.exists(config.METADATA_PATH):
         raise FileNotFoundError(f"Metadata file not found at {config.METADATA_PATH}. Please make sure data is placed correctly.")

    data = pd.read_csv(config.METADATA_PATH)
    data['image_path'] = os.path.join(config.IMAGE_DIR, '') + data['image_id'] + '.jpg'

    data['cell_type'] = data['dx'].map(config.LESION_TYPE_DICT)
    data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

    data['age'] = data['age'].fillna(data['age'].mean())

    print("Splitting dataset...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['cell_type_idx'])
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['cell_type_idx'])

    # 3. Apply Oversampling (RandomOverSampler) to Training Data
    if apply_smote:
        print("Applying RandomOverSampler to balance training classes...")
        # Using RandomOverSampler for image paths is safer for raw images than SMOTE interpolating pixels.
        ros = RandomOverSampler(random_state=42)
        X_train_paths = train_data['image_path'].values.reshape(-1, 1)
        y_train_labels = train_data['cell_type_idx'].values

        X_resampled, y_resampled = ros.fit_resample(X_train_paths, y_train_labels)

        train_paths = X_resampled.flatten()
        train_labels = y_resampled
    else:
        train_paths = train_data['image_path'].values
        train_labels = train_data['cell_type_idx'].values

    val_paths = val_data['image_path'].values
    val_labels = val_data['cell_type_idx'].values

    test_paths = test_data['image_path'].values
    test_labels = test_data['cell_type_idx'].values

    # 4. Create tf.data.Dataset objects (Lazy Loading & Augmentation)
    print("Creating tf.data.Datasets...")
    print(f"Training samples (after oversampling): {len(train_paths)}")

    train_dataset = create_tf_dataset(train_paths, train_labels, is_training=True)
    val_dataset = create_tf_dataset(val_paths, val_labels, is_training=False)
    test_dataset = create_tf_dataset(test_paths, test_labels, is_training=False)

    return train_dataset, val_dataset, test_dataset
