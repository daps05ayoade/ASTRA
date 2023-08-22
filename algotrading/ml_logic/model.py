import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from colorama import Fore, Style

class TimeSeriesModel:
    def __init__(self, input_shape):
        self.model = self.initialize_model(input_shape)

    def initialize_model(self, input_shape):
        """
        Initialize the Neural Network
        """
        print(Fore.BLUE + "\nInitialize model..." + Style.RESET_ALL)

        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='linear'))

        print("\n✅ model initialized")

        return model

    def compile(self):
        """
        Compile the Neural Netwrok
        """
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print("\n✅ model compiled")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, batch_size, epochs):
        """
        Fit model and return
        """
        print(Fore.BLUE + "\nTrain model..." + Style.RESET_ALL)

        # Checkpoint callback to save best model during training
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_mae', save_best_only=True, verbose=1, options=None)

        # Early stopping callback to stop training when Mean Absolute Error stops improving
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)

        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

        print(f"\n✅ model trained ({len(X_train)} rows)")


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, batch_size=15):
        """
        Evaluate trained model performance on dataset
        """
        print(Fore.BLUE + f"\nEvaluate model on {len(X_test)} rows..." + Style.RESET_ALL)

        if self.model is None:
            print(f"\n❌ no model to evaluate")
            return None

        metrics = self.model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, return_dict=True)
        y_pred = self.model.predict(X_test).flatten()

        loss = metrics["loss"]
        mae = metrics["mae"]

        print(f"\n✅ model evaluated: loss {round(loss, 2)} mae {round(mae, 2)}")

        return y_pred
