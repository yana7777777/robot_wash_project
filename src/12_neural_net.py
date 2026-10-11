"""
12_neural_net.py — Нейронная сеть (MLP, Keras/TensorFlow).

Задача: регрессия — прогноз прибыли по месяцам.

Читает:  data/robot_wash_monthly.csv
Пишет:   models/neural_net.h5
         reports/neural_net_report.txt
         images/51-54*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

MONTHLY_PATH = 'data/robot_wash_monthly.csv'
MODELS_DIR = 'models'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# ШАГ 1. ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("ШАГ 1. Загрузка данных")
print("=" * 60)

df = pd.read_csv(MONTHLY_PATH)
feature_cols = [
    'month', 'avg_exchange_rate', 'robot_sales_count',
    'service_count', 'import_count', 'transactions',
    'income_lag1', 'profit_lag1', 'income_lag3', 'profit_rolling3'
]
target = 'profit'

df_model = df[feature_cols + [target]].dropna()
X = df_model[feature_cols].values
y = df_model[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
)
print(f"Обучающая: {len(X_train)}, тестовая: {len(X_test)}")


# ============================================================
# ШАГ 2. МОДЕЛЬ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Построение MLP")
print("=" * 60)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-5)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=300, batch_size=8,
    callbacks=[es, rlr],
    verbose=0
)
print(f"Обучение завершено, эпох: {len(history.history['loss'])}")


# ============================================================
# ШАГ 3. ОЦЕНКА
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Оценка")
print("=" * 60)

y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²:   {r2:.4f}")


# ============================================================
# ШАГ 4. ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 51. Кривые обучения
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val', color='red')
axes[0].set_title('Loss (MSE)')
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['mae'], label='Train', color='steelblue')
axes[1].plot(history.history['val_mae'], label='Val', color='red')
axes[1].set_title('MAE')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/51_nn_learning.png', dpi=150)
plt.close()
print("Сохранён: 51_nn_learning.png")

# 52. Реальные vs прогноз
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_true)), y_true, marker='o', label='Реальные',
         color='green', linewidth=2)
plt.plot(range(len(y_pred)), y_pred, marker='s', label='Прогноз MLP',
         color='red', linewidth=2)
plt.title(f'MLP: прогноз прибыли (R² = {r2:.4f})')
plt.xlabel('Месяц (тест)')
plt.ylabel('Прибыль, руб.')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/52_nn_forecast.png', dpi=150)
plt.close()
print("Сохранён: 52_nn_forecast.png")

# 53. Остатки
residuals = y_true - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.6, color='purple')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Остатки')
axes[0].set_xlabel('Прогноз')
axes[0].set_ylabel('Остатки')

axes[1].hist(residuals, bins=15, color='skyblue', edgecolor='black')
axes[1].set_title('Распределение остатков')
axes[1].set_xlabel('Остатки')
axes[1].set_ylabel('Частота')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/53_nn_residuals.png', dpi=150)
plt.close()
print("Сохранён: 53_nn_residuals.png")

# 54. Архитектура (текстом)
plt.figure(figsize=(10, 6))
plt.text(0.5, 0.5,
         f"MLP Architecture:\n\n"
         f"Input:  {X_train.shape[1]} признаков\n"
         f"Dense(64) + BatchNorm + Dropout(0.2)\n"
         f"Dense(32) + Dropout(0.2)\n"
         f"Dense(16)\n"
         f"Dense(1) — выход\n\n"
         f"Params: {model.count_params():,}\n"
         f"Optimizer: Adam (lr=0.001)\n"
         f"Loss: MSE\n"
         f"Epochs run: {len(history.history['loss'])}\n"
         f"R²: {r2:.4f}",
         ha='center', va='center', fontsize=12, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.axis('off')
plt.title('Архитектура нейронной сети')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/54_nn_architecture.png', dpi=150)
plt.close()
print("Сохранён: 54_nn_architecture.png")


# ============================================================
# ШАГ 5. СОХРАНЕНИЕ
# ============================================================
model.save(f'{MODELS_DIR}/neural_net.h5')
print(f"\nМодель сохранена: {MODELS_DIR}/neural_net.h5")


# ============================================================
# ОТЧЁТ
# ============================================================
with open(f'{REPORTS_DIR}/neural_net_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: НЕЙРОННАЯ СЕТЬ (MLP)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Признаков: {X_train.shape[1]}\n")
    f.write(f"Обучающая: {len(X_train)}, тестовая: {len(X_test)}\n")
    f.write(f"Параметров модели: {model.count_params():,}\n")
    f.write(f"Эпох: {len(history.history['loss'])}\n\n")
    f.write(f"MAE:  {mae:,.0f} руб.\n")
    f.write(f"RMSE: {rmse:,.0f} руб.\n")
    f.write(f"R²:   {r2:.4f}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/neural_net_report.txt")


print("\n" + "=" * 60)
print("ГОТОВО! Нейросеть завершена.")
print("=" * 60)