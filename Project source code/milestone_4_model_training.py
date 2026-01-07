import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc # Garbage Collector for memory management
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("--- STARTING MILESTONE 4: PROFESSIONAL MODEL TRAINING ---")

# 1. Load the FINAL Dataset
print("1. Loading Data...")
df = pd.read_csv('8_FINAL_TRAINING_DATA_V2.csv')

# Optimization: Convert to float32 to save 50% RAM immediately
# The AI doesn't need 64-bit precision; 32-bit is standard for Deep Learning.
X_data = df.drop(columns=['market_value']).values.astype('float32')
y_data = df['market_value'].values.reshape(-1, 1).astype('float32')

# Free up the dataframe memory
del df
gc.collect()

# 2. Scale the Target (Market Value)
# AI works best when outputs are between 0 and 1
print("2. Scaling Target...")
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_data)

# 3. Create Time-Series Sequences
# We look back 3 steps to predict the 4th step.
TIME_STEPS = 3

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

print("3. Creating Sequences (This will take approx 1 minute)...")
X_seq, y_seq = create_sequences(X_data, y_scaled, TIME_STEPS)

# Free up raw data
del X_data, y_data
gc.collect()

print(f"   -> Input Shape: {X_seq.shape} (Rows, Time Steps, Features)")

# 4. Split Train vs. Test
# We use 80% for training (Past), 20% for testing (Future)
# We DO NOT shuffle, because order matters in time series!
split_idx = int(len(X_seq) * 0.8)

X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"   -> Training on {len(X_train)} samples")
print(f"   -> Testing on {len(X_test)} samples")

# 5. Build the LSTM Architecture
print("4. Building LSTM Architecture...")
model = Sequential()

# Layer 1: LSTM (The Brain)
# 64 Units is a robust size for this data.
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Prevents the AI from memorizing data (Overfitting)

# Layer 2: LSTM
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# Layer 3: Dense (The Output)
model.add(Dense(1)) # Predicts a single number (Market Value)

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Define Callbacks (The "Smart" features)
callbacks = [
    # Stop if validation loss doesn't improve for 3 epochs (saves time)
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
    # Save the BEST model found during training
    ModelCheckpoint('best_football_model.keras', monitor='val_loss', save_best_only=True)
]

# 7. Train the Model
print("5. Training Started (This may take 5-15 minutes)...")
# Batch Size 1024: A perfect balance between Speed and Accuracy for this dataset size.
history = model.fit(
    X_train, y_train,
    epochs=20,              # Max Limit (EarlyStopping will likely stop it sooner)
    batch_size=1024,        # Large batch size prevents PC freeze
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 8. Save Final Results
print("-> Training Complete.")
model.save('final_football_model.h5')
print("-> Model saved as 'final_football_model.h5'")

# 9. Plot the Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss (Error)')
plt.plot(history.history['val_loss'], label='Validation Loss (Test Error)')
plt.title('AI Training Performance')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.savefig('training_performance.png')
print("-> Performance Chart saved as 'training_performance.png'")