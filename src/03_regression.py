"""
03_regression.py — Регрессия: прогноз прибыли по месяцам.

Читает:  data/robot_wash_monthly.csv
Пишет:   models/best_regression.pkl
         reports/regression_report.txt
         images/08-12*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

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
print("ШАГ 1. Загрузка помесячных данных")
print("=" * 60)

df = pd.read_csv(MONTHLY_PATH)
print(f"Строк: {len(df)}")
print(f"Колонок: {len(df.columns)}")


# ============================================================
# ШАГ 2. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Подготовка признаков")
print("=" * 60)

feature_cols = [
    'month', 'avg_exchange_rate', 'robot_sales_count',
    'service_count', 'import_count', 'transactions',
    'income_lag1', 'profit_lag1', 'income_lag3', 'profit_rolling3'
]

target = 'profit'

df_model = df[feature_cols + [target, 'year_month']].dropna()
print(f"Строк после удаления пропусков: {len(df_model)}")

X = df_model[feature_cols].values
y = df_model[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)
print(f"Обучающая выборка: {len(X_train)}")
print(f"Тестовая выборка:  {len(X_test)}")


# ============================================================
# ШАГ 3. ОБУЧЕНИЕ 12 МОДЕЛЕЙ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Обучение 12 моделей")
print("=" * 60)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(n_estimators=200, random_state=42, verbose=0),
    'SVR': SVR(kernel='rbf', C=1000),
}

results = []
predictions = {}

for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Модель': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
        predictions[name] = y_pred

        print(f"{name:22s} | MAE: {mae:>14,.0f} | RMSE: {rmse:>14,.0f} | R²: {r2:.4f}")
    except Exception as e:
        print(f"{name:22s} | ОШИБКА: {e}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print("\nТОП-3 модели:")
print(results_df.head(3).to_string(index=False))

best_model_name = results_df.iloc[0]['Модель']
print(f"\nЛучшая модель: {best_model_name}")


# ============================================================
# ШАГ 4. ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Построение графиков")
print("=" * 60)

# 4.1 Реальные vs предсказанные (лучшая модель)
best_pred = predictions[best_model_name]

plt.figure(figsize=(14, 6))
plt.plot(range(len(y_test)), y_test, marker='o', label='Реальные', color='green', linewidth=2)
plt.plot(range(len(y_test)), best_pred, marker='s', label=f'Прогноз ({best_model_name})', color='red', linewidth=2)
plt.title(f'Прогноз прибыли: реальные vs предсказанные ({best_model_name})')
plt.xlabel('Месяц (тестовая выборка)')
plt.ylabel('Прибыль, руб.')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/08_regression_best.png', dpi=150)
plt.close()
print("Сохранён: 08_regression_best.png")

# 4.2 Сравнение моделей по R²
plt.figure(figsize=(12, 8))
colors = ['green' if x == results_df.iloc[0]['R²'] else 'steelblue' for x in results_df['R²']]
plt.barh(results_df['Модель'], results_df['R²'], color=colors)
plt.title('Сравнение моделей регрессии по R² (чем больше, тем лучше)')
plt.xlabel('R²')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/09_regression_r2.png', dpi=150)
plt.close()
print("Сохранён: 09_regression_r2.png")

# 4.3 Сравнение по MAE
plt.figure(figsize=(12, 8))
plt.barh(results_df['Модель'], results_df['MAE'], color='coral')
plt.title('Сравнение моделей по MAE (чем меньше, тем лучше)')
plt.xlabel('MAE, руб.')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/10_regression_mae.png', dpi=150)
plt.close()
print("Сохранён: 10_regression_mae.png")

# 4.4 Важность признаков (Random Forest)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx], color='teal')
plt.title('Важность признаков (Random Forest)')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/11_regression_importance.png', dpi=150)
plt.close()
print("Сохранён: 11_regression_importance.png")

# 4.5 Остатки лучшей модели
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(best_pred, residuals, alpha=0.6, color='purple')
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_title('Остатки лучшей модели')
axes[0].set_xlabel('Предсказанные значения')
axes[0].set_ylabel('Остатки')

axes[1].hist(residuals, bins=20, color='skyblue', edgecolor='black')
axes[1].set_title('Распределение остатков')
axes[1].set_xlabel('Остатки')
axes[1].set_ylabel('Частота')

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/12_regression_residuals.png', dpi=150)
plt.close()
print("Сохранён: 12_regression_residuals.png")


# ============================================================
# ШАГ 5. СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. Сохранение лучшей модели")
print("=" * 60)

best_model = models[best_model_name]
with open(f'{MODELS_DIR}/best_regression.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Модель сохранена: {MODELS_DIR}/best_regression.pkl")


# ============================================================
# ШАГ 6. ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 6. Отчёт")
print("=" * 60)

with open(f'{REPORTS_DIR}/regression_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: РЕГРЕССИЯ (прогноз прибыли по месяцам)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Обучающая выборка: {len(X_train)}\n")
    f.write(f"Тестовая выборка:  {len(X_test)}\n")
    f.write(f"Всего моделей:     {len(models)}\n\n")
    f.write("Результаты (отсортированы по R²):\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    f.write(f"Лучшая модель: {best_model_name}\n")
    f.write(f"R²:   {results_df.iloc[0]['R²']:.4f}\n")
    f.write(f"MAE:  {results_df.iloc[0]['MAE']:,.0f} руб.\n")
    f.write(f"RMSE: {results_df.iloc[0]['RMSE']:,.0f} руб.\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/regression_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Регрессия завершена.")
print("=" * 60)
print(f"Обучено моделей:   {len(models)}")
print(f"Лучшая модель:     {best_model_name}")
print(f"R² лучшей:         {results_df.iloc[0]['R²']:.4f}")