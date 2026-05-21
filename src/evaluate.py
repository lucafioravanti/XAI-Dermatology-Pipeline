import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import src.config as config
import tensorflow as tf

def plot_training_history(history_head, history_finetune):
    """Plots training and validation loss and accuracy for both phases."""
    plt.figure(figsize=(12, 10))

    # Combine histories for plotting
    loss = history_head.history['loss'] + history_finetune.history['loss']
    val_loss = history_head.history['val_loss'] + history_finetune.history['val_loss']
    acc = history_head.history['accuracy'] + history_finetune.history['accuracy']
    val_acc = history_head.history['val_accuracy'] + history_finetune.history['val_accuracy']

    epochs_range = range(1, len(loss) + 1)

    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    # Add a vertical line to indicate where fine-tuning starts
    plt.axvline(x=len(history_head.history['loss']), color='r', linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    plt.title('Training and Validation Loss (Two-Phase)')

    # Plot Accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.axvline(x=len(history_head.history['accuracy']), color='r', linestyle='--', label='Start Fine-Tuning')
    plt.legend()
    plt.title('Training and Validation Accuracy (Two-Phase)')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Saved training history plot to training_history.png")

def evaluate_model(model, test_dataset):
    """Evaluates the model on test dataset and prints metrics."""

    print("\nEvaluating on Test Data...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    print("\nGenerating Predictions...")
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)

    # Extract true labels from the tf.data.Dataset
    true_labels = []
    for images, labels in test_dataset.unbatch():
        true_labels.append(labels.numpy())
    true_labels = np.array(true_labels)

    print("\nConfusion Matrix:")
    confusion_mat = confusion_matrix(true_labels, predicted_labels)
    print(confusion_mat)

    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))

    # Visualize some predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(true_labels))):
        print(f"True Label: {true_labels[i]}, Predicted Label: {predicted_labels[i]}")
