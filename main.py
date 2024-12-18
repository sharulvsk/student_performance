from scripts.preprocess import load_and_preprocess
from scripts.ml_model import train_ml_model
from scripts.dl_model import train_dl_model
import matplotlib.pyplot as plt
import os

def plot_history(history, save_path="plots/plot_training_history.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)  

    print(f"Training history plot saved to {save_path}")

if __name__ == "__main__":
    filepath = r"C:\From Destop\student_performance\data\student_performance.csv"
    X_train, X_test, y_train, y_test = load_and_preprocess(filepath)
    print("Training ML Model...")
    ml_model = train_ml_model(X_train, y_train, X_test, y_test)
    print("Training DL Model...")
    dl_model, history = train_dl_model(X_train, y_train, X_test, y_test)
    print("Plotting Training History...")
    plot_history(history)

