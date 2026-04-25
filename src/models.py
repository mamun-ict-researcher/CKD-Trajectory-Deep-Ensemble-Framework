import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
from catboost import CatBoostRegressor

def build_lstm_model(input_shape, horizon):
    # Deep sequential modeling for temporal features
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=False, dropout=0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(horizon, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_transformer_model(input_shape, horizon):
    # Attention mechanism for global dependency modeling
    inputs = layers.Input(shape=input_shape)
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=input_shape[-1])(x, x)
    x = layers.Add()([x, attn])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(horizon, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def get_tree_models(random_state=42):
    # Tree ensembles for capturing static risk profiles
    xgb_m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state)
    cat_m = CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1, verbose=0, random_seed=random_state)
    return xgb_m, cat_m