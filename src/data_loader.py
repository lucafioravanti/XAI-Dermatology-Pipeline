import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import src.config as config

def process_path(file_path, label):
    """
    TensorFlow mapping function to load, resize, and normalize an image from a path.
    """
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    # Decode the image (works for JPG, PNG, etc.)
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, [config.IMAGE_SIZE, config.IMAGE_SIZE])
    # Normalize to [0, 1]
    img = img / 255.0
    return img, label

def create_tf_dataset(paths, labels, is_training=False):
    """
    Creates an optimized tf.data.Dataset pipeline.
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=len(paths))

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def load_and_preprocess_data(apply_oversampling=True):
    """
    Loads metadata, creates image paths, splits data, balances classes,
    and returns optimized tf.data.Dataset objects.
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

    if apply_oversampling:
        print("Applying Random Oversampling to balance training classes...")
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

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    print("Creating tf.data pipelines...")
    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_labels, is_training=False)

    return train_ds, val_ds, test_ds, test_labels # Return raw labels for confusion matrix
