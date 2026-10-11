"""
04_classification.py — Классификация транзакций.

Задача 1 (бинарная): доход / расход.
Задача 2 (многоклассовая): статья дохода.

Читает:  data/robot_wash_features.csv
Пишет:   models/best_classifier_binary.pkl
         models/best_classifier_multi.pkl
         reports/classification_report.txt
         images/13-20*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report as sk_class_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

FEATURES_PATH = 'data/robot_wash_features.csv'
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
print("ШАГ 1. Загрузка признаков")
print("=" * 60)

df = pd.read_csv(FEATURES_PATH, dtype={'tnved_code': str}, parse_dates=['date'])
print(f"Строк: {len(df)}")
print(f"Колонок: {len(df.columns)}")


# ============================================================
# ЗАДАЧА 1. БИНАРНАЯ КЛАССИФИКАЦИЯ: ДОХОД / РАСХОД
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 1. Бинарная классификация: доход / расход")
print("=" * 70)

feature_cols_bin = [
    'quantity', 'exchange_rate', 'customs_duty_rate',
    'customs_duty_amount', 'vat_rate', 'vat_amount',
    'customs_fee', 'bank_commission', 'delivery_cost_rub',
    'month', 'day_of_week', 'quarter_num', 'is_import'
]

df_bin = df[feature_cols_bin + ['is_income']].dropna()
X_bin = df_bin[feature_cols_bin].values
y_bin = df_bin['is_income'].values

print(f"Строк: {len(df_bin)}")
print(f"Класс 0 (расход): {(y_bin == 0).sum()}")
print(f"Класс 1 (доход):  {(y_bin == 1).sum()}")

# Масштабирование
scaler_bin = StandardScaler()
X_bin_scaled = scaler_bin.fit_transform(X_bin)

X_train, X_test, y_train, y_test = train_test_split(
    X_bin_scaled, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
print(f"Обучающая: {len(X_train)}, тестовая: {len(X_test)}")


# Модели
models_bin = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, verbosity=0, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(n_estimators=200, random_state=42, verbose=0),
}

results_bin = []

print("\nОбучение моделей (бинарная):")
for name, model in models_bin.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0

    results_bin.append({
        'Модель': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1, 'ROC-AUC': auc
    })

    print(f"{name:22s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

results_bin_df = pd.DataFrame(results_bin).sort_values('F1', ascending=False)
best_bin_name = results_bin_df.iloc[0]['Модель']
print(f"\nЛучшая (бинарная): {best_bin_name} (F1 = {results_bin_df.iloc[0]['F1']:.4f})")

# Сохранение лучшей модели
with open(f'{MODELS_DIR}/best_classifier_binary.pkl', 'wb') as f:
    pickle.dump(models_bin[best_bin_name], f)


# ============================================================
# ЗАДАЧА 2. МНОГОКЛАССОВАЯ: СТАТЬЯ ДОХОДА
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 2. Многоклассовая классификация: статья дохода")
print("=" * 70)

df_multi = df[df['is_income'] == 1].copy()
df_multi = df_multi[df_multi['income_article'] != '']

feature_cols_multi = [
    'quantity', 'exchange_rate', 'customs_duty_rate',
    'customs_duty_amount', 'vat_rate', 'vat_amount',
    'month', 'day_of_week', 'quarter_num'
]

df_multi = df_multi[feature_cols_multi + ['income_article']].dropna()
X_multi = df_multi[feature_cols_multi].values
y_multi_raw = df_multi['income_article'].values

le = LabelEncoder()
y_multi = le.fit_transform(y_multi_raw)

print(f"Строк: {len(df_multi)}")
print(f"Классов: {len(le.classes_)}")
for cls in le.classes_:
    print(f"  - {cls}")

scaler_multi = StandardScaler()
X_multi_scaled = scaler_multi.fit_transform(X_multi)

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi_scaled, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)
print(f"Обучающая: {len(X_train_m)}, тестовая: {len(X_test_m)}")

models_multi = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, verbosity=0, eval_metric='mlogloss'),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(n_estimators=200, random_state=42, verbose=0),
}

results_multi = []

print("\nОбучение моделей (многоклассовая):")
for name, model in models_multi.items():
    model.fit(X_train_m, y_train_m)
    y_pred_m = model.predict(X_test_m)

    acc = accuracy_score(y_test_m, y_pred_m)
    f1 = f1_score(y_test_m, y_pred_m, average='weighted', zero_division=0)
    prec = precision_score(y_test_m, y_pred_m, average='weighted', zero_division=0)
    rec = recall_score(y_test_m, y_pred_m, average='weighted', zero_division=0)

    results_multi.append({
        'Модель': name, 'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1': f1
    })

    print(f"{name:22s} | Acc: {acc:.4f} | F1: {f1:.4f}")

results_multi_df = pd.DataFrame(results_multi).sort_values('F1', ascending=False)
best_multi_name = results_multi_df.iloc[0]['Модель']
print(f"\nЛучшая (многоклассовая): {best_multi_name} (F1 = {results_multi_df.iloc[0]['F1']:.4f})")

with open(f'{MODELS_DIR}/best_classifier_multi.pkl', 'wb') as f:
    pickle.dump(models_multi[best_multi_name], f)


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 13. Сравнение моделей (бинарная) по F1
plt.figure(figsize=(12, 6))
plt.barh(results_bin_df['Модель'], results_bin_df['F1'], color='steelblue')
plt.title('Бинарная классификация: сравнение моделей по F1')
plt.xlabel('F1-score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/13_class_binary_f1.png', dpi=150)
plt.close()
print("Сохранён: 13_class_binary_f1.png")

# 14. Матрица ошибок (лучшая бинарная)
best_bin_model = models_bin[best_bin_name]
y_pred_best = best_bin_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Расход', 'Доход'],
            yticklabels=['Расход', 'Доход'])
plt.title(f'Матрица ошибок ({best_bin_name})')
plt.ylabel('Истина')
plt.xlabel('Прогноз')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/14_class_binary_cm.png', dpi=150)
plt.close()
print("Сохранён: 14_class_binary_cm.png")

# 15. Сравнение моделей (многоклассовая)
plt.figure(figsize=(12, 6))
plt.barh(results_multi_df['Модель'], results_multi_df['F1'], color='teal')
plt.title('Многоклассовая классификация: сравнение моделей по F1')
plt.xlabel('F1-score (weighted)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/15_class_multi_f1.png', dpi=150)
plt.close()
print("Сохранён: 15_class_multi_f1.png")

# 16. Матрица ошибок (лучшая многоклассовая)
best_multi_model = models_multi[best_multi_name]
y_pred_m_best = best_multi_model.predict(X_test_m)
cm_m = confusion_matrix(y_test_m, y_pred_m_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_m, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Матрица ошибок ({best_multi_name})')
plt.ylabel('Истина')
plt.xlabel('Прогноз')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/16_class_multi_cm.png', dpi=150)
plt.close()
print("Сохранён: 16_class_multi_cm.png")

# 17. Важность признаков (Random Forest, бинарная)
rf_bin = models_bin['Random Forest']
importances = rf_bin.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.barh(np.array(feature_cols_bin)[sorted_idx], importances[sorted_idx], color='coral')
plt.title('Важность признаков (Random Forest, бинарная)')
plt.xlabel('Важность')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/17_class_binary_importance.png', dpi=150)
plt.close()
print("Сохранён: 17_class_binary_importance.png")


# ============================================================
# ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

with open(f'{REPORTS_DIR}/classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: КЛАССИФИКАЦИЯ ТРАНЗАКЦИЙ\n")
    f.write("=" * 70 + "\n\n")

    f.write("ЗАДАЧА 1. Бинарная: доход / расход\n")
    f.write("-" * 70 + "\n")
    f.write(f"Строк: {len(df_bin)}\n")
    f.write(f"Обучающая: {len(X_train)}, тестовая: {len(X_test)}\n\n")
    f.write(results_bin_df.to_string(index=False))
    f.write(f"\n\nЛучшая модель: {best_bin_name}\n")
    f.write(f"F1: {results_bin_df.iloc[0]['F1']:.4f}\n")
    f.write(f"Accuracy: {results_bin_df.iloc[0]['Accuracy']:.4f}\n\n")

    f.write("\nЗАДАЧА 2. Многоклассовая: статья дохода\n")
    f.write("-" * 70 + "\n")
    f.write(f"Строк: {len(df_multi)}\n")
    f.write(f"Классов: {len(le.classes_)}\n")
    f.write(f"Классы: {', '.join(le.classes_)}\n\n")
    f.write(results_multi_df.to_string(index=False))
    f.write(f"\n\nЛучшая модель: {best_multi_name}\n")
    f.write(f"F1: {results_multi_df.iloc[0]['F1']:.4f}\n")
    f.write(f"Accuracy: {results_multi_df.iloc[0]['Accuracy']:.4f}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/classification_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Классификация завершена.")
print("=" * 60)
print(f"Бинарная — лучшая: {best_bin_name}")
print(f"Многоклассовая — лучшая: {best_multi_name}")