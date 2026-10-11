"""
11_stacking.py — Стекинг и ансамбли моделей (Voting, Bagging, Boosting, Stacking).

Читает:  data/robot_wash_monthly.csv
Пишет:   models/stacking_model.pkl
         reports/stacking_report.txt
         images/47-50*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              VotingRegressor, BaggingRegressor, StackingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
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
# ШАГ 2. АНСАМБЛИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Обучение ансамблей")
print("=" * 60)

base_lr = LinearRegression()
base_rf = RandomForestRegressor(n_estimators=200, random_state=42)
base_gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
base_xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)

models = {
    'Single LR': base_lr,
    'Single RF': base_rf,
    'Voting': VotingRegressor([('lr', LinearRegression()),
                                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))]),
    'Bagging': BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8),
                                 n_estimators=50, random_state=42),
    'Stacking': StackingRegressor(
        estimators=[('lr', LinearRegression()),
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('xgb', XGBRegressor(n_estimators=100, random_state=42, verbosity=0))],
        final_estimator=Ridge(alpha=1.0)
    ),
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({'Модель': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2})
    predictions[name] = y_pred
    print(f"{name:15s} | MAE: {mae:>14,.0f} | RMSE: {rmse:>14,.0f} | R²: {r2:.4f}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
best_name = results_df.iloc[0]['Модель']
print(f"\nЛучший ансамбль: {best_name}")


# ============================================================
# ШАГ 3. ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 47. Сравнение R² ансамблей
plt.figure(figsize=(10, 6))
colors = ['green' if x == results_df.iloc[0]['R²'] else 'steelblue' for x in results_df['R²']]
plt.barh(results_df['Модель'], results_df['R²'], color=colors)
plt.title('Сравнение ансамблей по R²')
plt.xlabel('R²')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/47_stacking_r2.png', dpi=150)
plt.close()
print("Сохранён: 47_stacking_r2.png")

# 48. Реальные vs прогноз (лучший)
plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test)), y_test, marker='o', label='Реальные',
         color='green', linewidth=2)
plt.plot(range(len(y_test)), predictions[best_name], marker='s',
         label=f'Прогноз ({best_name})', color='red', linewidth=2)
plt.title(f'Лучший ансамбль: {best_name}')
plt.xlabel('Месяц (тестовая выборка)')
plt.ylabel('Прибыль, руб.')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/48_stacking_best.png', dpi=150)
plt.close()
print("Сохранён: 48_stacking_best.png")

# 49. MAE сравнение
plt.figure(figsize=(10, 6))
plt.barh(results_df['Модель'], results_df['MAE'], color='coral')
plt.title('Сравнение ансамблей по MAE')
plt.xlabel('MAE, руб.')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/49_stacking_mae.png', dpi=150)
plt.close()
print("Сохранён: 49_stacking_mae.png")

# 50. Все прогнозы
plt.figure(figsize=(14, 7))
plt.plot(range(len(y_test)), y_test, marker='o', label='Реальные',
         color='black', linewidth=2.5)
for name, pred in predictions.items():
    plt.plot(range(len(y_test)), pred, marker='.', label=name, alpha=0.6)
plt.title('Сравнение прогнозов всех ансамблей')
plt.xlabel('Месяц')
plt.ylabel('Прибыль, руб.')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/50_stacking_all.png', dpi=150)
plt.close()
print("Сохранён: 50_stacking_all.png")


# ============================================================
# ШАГ 4. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================================
best_model = models[best_name]
with open(f'{MODELS_DIR}/stacking_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nМодель сохранена: {MODELS_DIR}/stacking_model.pkl")


# ============================================================
# ОТЧЁТ
# ============================================================
with open(f'{REPORTS_DIR}/stacking_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: АНСАМБЛИ И СТЕКИНГ\n")
    f.write("=" * 60 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nЛучший ансамбль: {best_name}\n")
    f.write(f"R²:   {results_df.iloc[0]['R²']:.4f}\n")
    f.write(f"MAE:  {results_df.iloc[0]['MAE']:,.0f} руб.\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/stacking_report.txt")


print("\n" + "=" * 60)
print("ГОТОВО! Стекинг завершён.")
print("=" * 60)