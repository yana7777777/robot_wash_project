"""
14_optuna_tuning.py — Байесовская оптимизация гиперпараметров (Optuna).

Задача: найти лучшие гиперпараметры для XGBoost/LightGBM на прогнозе прибыли.

Читает:  data/robot_wash_monthly.csv
Пишет:   models/best_optuna_model.pkl
         reports/optuna_report.txt
         images/59-62*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
print(f"Обучающая: {len(X_train)}, тестовая: {len(X_test)}")


# ============================================================
# ШАГ 2. OPTUNA — XGBOOST
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Optuna: XGBoost")
print("=" * 60)

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y_train,
                              cv=3, scoring='neg_mean_absolute_error')
    return -scores.mean()

study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)

print(f"Лучший MAE (CV): {study_xgb.best_value:,.0f}")
print(f"Лучшие параметры:")
for k, v in study_xgb.best_params.items():
    print(f"  {k}: {v}")

best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"\nXGBoost (Optuna): R² = {r2_xgb:.4f}, MAE = {mae_xgb:,.0f}")


# ============================================================
# ШАГ 3. OPTUNA — LIGHTGBM
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Optuna: LightGBM")
print("=" * 60)

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 2, 30),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
        'verbose': -1
    }
    model = LGBMRegressor(**params)
    scores = cross_val_score(model, X_train, y_train,
                              cv=3, scoring='neg_mean_absolute_error')
    return -scores.mean()

study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=False)

print(f"Лучший MAE (CV): {study_lgb.best_value:,.0f}")
print(f"Лучшие параметры:")
for k, v in study_lgb.best_params.items():
    print(f"  {k}: {v}")

best_lgb = LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
best_lgb.fit(X_train, y_train)
y_pred_lgb = best_lgb.predict(X_test)
r2_lgb = r2_score(y_test, y_pred_lgb)
mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
print(f"\nLightGBM (Optuna): R² = {r2_lgb:.4f}, MAE = {mae_lgb:,.0f}")


# ============================================================
# ШАГ 4. ВЫБОР ЛУЧШЕЙ МОДЕЛИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Сравнение")
print("=" * 60)

if r2_xgb >= r2_lgb:
    best_model = best_xgb
    best_name = 'XGBoost (Optuna)'
    best_pred = y_pred_xgb
    best_r2 = r2_xgb
    best_mae = mae_xgb
    best_study = study_xgb
else:
    best_model = best_lgb
    best_name = 'LightGBM (Optuna)'
    best_pred = y_pred_lgb
    best_r2 = r2_lgb
    best_mae = mae_lgb
    best_study = study_lgb

print(f"Лучшая модель: {best_name}")
print(f"R²: {best_r2:.4f}")


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 59. История оптимизации XGBoost
trials_xgb = [t.value for t in study_xgb.trials if t.value is not None]
plt.figure(figsize=(12, 6))
plt.plot(trials_xgb, marker='o', alpha=0.5, color='steelblue', label='Триалы')
plt.plot(np.minimum.accumulate(trials_xgb), color='red', linewidth=2, label='Лучший')
plt.title('Optuna: XGBoost — история оптимизации')
plt.xlabel('Триал')
plt.ylabel('MAE (CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/59_optuna_xgb_history.png', dpi=150)
plt.close()
print("Сохранён: 59_optuna_xgb_history.png")

# 60. История оптимизации LightGBM
trials_lgb = [t.value for t in study_lgb.trials if t.value is not None]
plt.figure(figsize=(12, 6))
plt.plot(trials_lgb, marker='o', alpha=0.5, color='coral', label='Триалы')
plt.plot(np.minimum.accumulate(trials_lgb), color='red', linewidth=2, label='Лучший')
plt.title('Optuna: LightGBM — история оптимизации')
plt.xlabel('Триал')
plt.ylabel('MAE (CV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/60_optuna_lgb_history.png', dpi=150)
plt.close()
print("Сохранён: 60_optuna_lgb_history.png")

# 61. Сравнение моделей
plt.figure(figsize=(10, 6))
models_names = ['XGBoost', 'LightGBM']
r2_scores = [r2_xgb, r2_lgb]
colors = ['green' if v == max(r2_scores) else 'steelblue' for v in r2_scores]
bars = plt.bar(models_names, r2_scores, color=colors)
for bar, v in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
             f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
plt.title('Optuna: сравнение моделей по R²')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/61_optuna_compare.png', dpi=150)
plt.close()
print("Сохранён: 61_optuna_compare.png")

# 62. Реальные vs прогноз лучшей
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test)), y_test, marker='o', label='Реальные',
         color='green', linewidth=2)
plt.plot(range(len(y_test)), best_pred, marker='s',
         label=f'Прогноз ({best_name})', color='red', linewidth=2)
plt.title(f'{best_name}: R² = {best_r2:.4f}')
plt.xlabel('Месяц (тест)')
plt.ylabel('Прибыль, руб.')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/62_optuna_forecast.png', dpi=150)
plt.close()
print("Сохранён: 62_optuna_forecast.png")


# ============================================================
# СОХРАНЕНИЕ МОДЕЛИ
# ============================================================
with open(f'{MODELS_DIR}/best_optuna_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nМодель сохранена: {MODELS_DIR}/best_optuna_model.pkl")


# ============================================================
# ОТЧЁТ
# ============================================================
with open(f'{REPORTS_DIR}/optuna_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ (Optuna)\n")
    f.write("=" * 70 + "\n\n")

    f.write("XGBoost:\n")
    f.write(f"  Триалов: {len(study_xgb.trials)}\n")
    f.write(f"  Лучший MAE (CV): {study_xgb.best_value:,.0f}\n")
    f.write(f"  Лучшие параметры:\n")
    for k, v in study_xgb.best_params.items():
        f.write(f"    {k}: {v}\n")
    f.write(f"  Test R²: {r2_xgb:.4f}, MAE: {mae_xgb:,.0f}\n\n")

    f.write("LightGBM:\n")
    f.write(f"  Триалов: {len(study_lgb.trials)}\n")
    f.write(f"  Лучший MAE (CV): {study_lgb.best_value:,.0f}\n")
    f.write(f"  Лучшие параметры:\n")
    for k, v in study_lgb.best_params.items():
        f.write(f"    {k}: {v}\n")
    f.write(f"  Test R²: {r2_lgb:.4f}, MAE: {mae_lgb:,.0f}\n\n")

    f.write(f"ЛУЧШАЯ МОДЕЛЬ: {best_name}\n")
    f.write(f"  R²:  {best_r2:.4f}\n")
    f.write(f"  MAE: {best_mae:,.0f} руб.\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/optuna_report.txt")


print("\n" + "=" * 60)
print("ГОТОВО! Optuna завершена.")
print("=" * 60)
print(f"Лучшая модель: {best_name}")
print(f"R²: {best_r2:.4f}")