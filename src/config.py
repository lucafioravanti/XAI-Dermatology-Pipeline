import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data') # Update to local data directory
METADATA_PATH = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR = os.path.join(DATA_DIR, 'HAM10000_all_images')

# Model parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 7

# Lesion dictionary
LESION_TYPE_DICT = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
