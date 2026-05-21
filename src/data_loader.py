import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import src.config as config

def preprocess_image(image_path, target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE)):
    """Loads, resizes, and normalizes an image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img) / 255.0
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return np.zeros((*target_size, 3)) # Return empty image on failure

def load_and_preprocess_data(apply_smote=True):
    """
    Loads metadata, creates image paths, splits data, applies SMOTE,
    and loads images into numpy arrays.
    """
    # 1. Load Metadata
    print("Loading metadata...")
    if not os.path.exists(config.METADATA_PATH):
         raise FileNotFoundError(f"Metadata file not found at {config.METADATA_PATH}. Please make sure data is placed correctly.")

    data = pd.read_csv(config.METADATA_PATH)
    data['image_path'] = os.path.join(config.IMAGE_DIR, '') + data['image_id'] + '.jpg'

    # Map labels to numeric indices
    data['cell_type'] = data['dx'].map(config.LESION_TYPE_DICT)
    data['cell_type_idx'] = pd.Categorical(data['cell_type']).codes

    # Fill missing ages
    data['age'] = data['age'].fillna(data['age'].mean())

    # 2. Initial Data Split (Train/Val/Test)
    print("Splitting dataset...")
    # Stratify by cell_type_idx to ensure distribution is maintained in splits before SMOTE
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['cell_type_idx'])
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42, stratify=train_data['cell_type_idx'])

    # 3. Apply SMOTE to Training Data
    if apply_smote:
        print("Applying SMOTE to balance training classes...")
        smote = SMOTE(random_state=42)

        # We need numerical features for SMOTE.
        # For images, doing SMOTE directly on 224x224x3 arrays requires a lot of RAM.
        # Alternatively, we can SMOTE the latent features, or just oversample the dataframe rows.
        # Given the previous EDA used SMOTE on tabular features, and we need images:
        # A common approach for image pipelines with SMOTE is to apply SMOTE on extracted features,
        # or simply duplicate image paths (random oversampling) if SMOTE on raw pixels OOMs.
        # Let's use SMOTE on a flattened proxy or just do Random Oversampling of the indices.
        # Actually, standard SMOTE generates *new* feature vectors.
        # If we SMOTE raw pixels: it interpolates pixels, which can create blurry/ghost images.
        # For this refactored pipeline, we will apply SMOTE on tabular data and duplicate images
        # or apply SMOTE on a latent representation later.
        # To strictly follow the EDA approach (which SMOTE'd tabular metadata):

        from imblearn.over_sampling import RandomOverSampler
        # Using RandomOverSampler for image paths is safer for raw images than interpolating pixels.
        ros = RandomOverSampler(random_state=42)

        # Reshape for the sampler
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

    # 4. Load Images into Memory (Note: For very large datasets, use a Generator/tf.data.Dataset instead)
    print("Loading and preprocessing images into memory...")
    print(f"Training samples: {len(train_paths)}")

    train_images = np.array([preprocess_image(p) for p in train_paths])
    val_images = np.array([preprocess_image(p) for p in val_paths])
    test_images = np.array([preprocess_image(p) for p in test_paths])

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)
