import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from src.data_loader import load_and_preprocess_data
from src.model import build_vit_model
from src.train import train_model
from src.evaluate import plot_training_history, evaluate_model

def main():
    print("=== XAI Dermatology Pipeline ===")

    # 1. Load and preprocess data (now returning tf.data.Dataset)
    try:
        train_dataset, val_dataset, test_dataset = load_and_preprocess_data(apply_smote=True)
    except FileNotFoundError as e:
        print(f"Dataset missing: {e}")
        print("Please ensure HAM10000_metadata.csv and HAM10000_all_images/ are in the data/ directory.")
        return

    # 2. Build model
    model = build_vit_model()
    model.summary()

    # 3. Train model (Two-Phase Fine-Tuning)
    history_head, history_finetune = train_model(model, train_dataset, val_dataset)

    # 4. Evaluate model
    plot_training_history(history_head, history_finetune)
    evaluate_model(model, test_dataset)

if __name__ == "__main__":
    main()