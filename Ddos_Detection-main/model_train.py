# model_train.py
import os
import json
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

def build_autoencoder(input_dim, encoding_dim=16):
    """
    Enhanced autoencoder with dropout and batch normalization for better DDoS detection.
    - input_dim: number of features
    - encoding_dim: size of bottleneck layer
    """
    from tensorflow.keras.layers import BatchNormalization, Dropout
    
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(64, activation="relu")(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Bottleneck
    encoded = Dense(encoding_dim, activation="relu")(x)
    
    # Decoder
    x = Dense(32, activation="relu")(encoded)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    
    # Output layer (sigmoid for normalized features)
    decoded = Dense(input_dim, activation="sigmoid")(x)
    
    # Autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder

def train_autoencoder_from_npy(npy_path,
                             model_dir="model",
                             results_dir="static/results",
                             epochs=100,  # Increased epochs for better convergence
                             batch_size=64,
                             encoding_dim=16,  # Increased bottleneck size
                             patience=10,  # Increased patience
                             validation_split=0.2,  # Added validation split
                             verbose=1):
    """
    Load X from npy, train autoencoder with improved settings, save model and training artifacts.
    Returns dict with paths and training info.
    """
    # Load and validate data
    X = np.load(npy_path)
    if len(X.shape) != 2:
        raise ValueError("X must be 2D array (n_samples, n_features)")

    # Ensure we have enough samples for training
    if X.shape[0] < 100:
        print("Warning: Small dataset may lead to poor model performance")

    input_dim = X.shape[1]

    # Ensure folders exist
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # build model
    model = build_autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)

    # callbacks (early stopping)
    early = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    history = model.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1 if X.shape[0] > 10 else 0.2,
        callbacks=[early],
        verbose=verbose,
        shuffle=True
    )

    # save model
    model_path = os.path.join(model_dir, "autoencoder.h5")
    model.save(model_path)

    # save history as json
    history_path = os.path.join(model_dir, "training_history.json")
    hist_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # save loss plot
    loss_fig_path = os.path.join(results_dir, "training_loss.png")
    plt.figure()
    plt.plot(hist_dict.get("loss", []))
    plt.plot(hist_dict.get("val_loss", []))
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Do not specify colors/styles (per instruction)
    plt.legend(["loss", "val_loss"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_fig_path)
    plt.close()

    return {
        "model_path": model_path,
        "history_path": history_path,
        "loss_plot": loss_fig_path,
        "epochs_ran": len(hist_dict.get("loss", [])),
        "final_train_loss": hist_dict.get("loss", [])[-1] if hist_dict.get("loss") else None,
        "final_val_loss": hist_dict.get("val_loss", [])[-1] if hist_dict.get("val_loss") else None
    }


def load_trained_autoencoder(model_dir="model"):
    path = os.path.join(model_dir, "autoencoder.h5")
    if not os.path.exists(path):
        raise FileNotFoundError("Trained model not found at: " + path)
    return load_model(path)
