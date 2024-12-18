from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_dl_model(X_train, y_train, X_test, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1)  
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
    return model, history

