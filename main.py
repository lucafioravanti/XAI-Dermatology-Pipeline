import os
from src.data_loader import load_and_preprocess_data
from src.model import build_vit_model
from src.train import train_model
from src.evaluate import plot_training_history, evaluate_model

def main():
    print("=== XAI Dermatology Pipeline ===")

    # 1. Load and preprocess data
    try:
        train_ds, val_ds, test_ds, test_labels = load_and_preprocess_data(apply_oversampling=True)
    except FileNotFoundError as e:
        print(f"Dataset missing: {e}")
        print("Please ensure HAM10000_metadata.csv and HAM10000_all_images/ are in the data/ directory.")
        return

    # 2. Build model
    model = build_vit_model()
    model.summary()

    # 3. Train model
    history = train_model(model, train_ds, val_ds)

    # 4. Evaluate model
    plot_training_history(history)
    evaluate_model(model, test_ds, test_labels)

if __name__ == "__main__":
    main()