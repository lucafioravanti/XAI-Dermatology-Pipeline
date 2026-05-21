import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import src.config as config

def plot_training_history(history):
    """Plots training and validation loss and accuracy."""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def evaluate_model(model, test_ds, test_labels):
    """Evaluates the model on test data and prints metrics."""

    print("\nEvaluating on Test Data...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    print("\nGenerating Predictions...")
    predictions = model.predict(test_ds)
    predicted_labels = np.argmax(predictions, axis=1)

    print("\nConfusion Matrix:")
    confusion_mat = confusion_matrix(test_labels, predicted_labels)
    print(confusion_mat)

    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_labels))

    # Visualize some predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(test_labels))):
        print(f"True Label: {test_labels[i]}, Predicted Label: {predicted_labels[i]}")
