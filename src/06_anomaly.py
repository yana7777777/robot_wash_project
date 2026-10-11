"""
06_anomaly.py — Обнаружение аномалий в финансовых операциях.

Методы: Isolation Forest, One-Class SVM, Local Outlier Factor.
Читает:  data/robot_wash_features.csv
Пишет:   reports/anomaly_report.txt
         reports/anomalies.csv
         images/24-27*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

FEATURES_PATH = 'data/robot_wash_features.csv'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

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

# Только транзакции с ненулевой суммой
df = df[df['total_amount'] > 0].copy().reset_index(drop=True)
print(f"Транзакций с суммой > 0: {len(df)}")


# ============================================================
# ШАГ 2. ПОДГОТОВКА ПРИЗНАКОВ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Подготовка признаков")
print("=" * 60)

feature_cols = [
    'total_amount', 'quantity', 'exchange_rate',
    'customs_duty_amount', 'vat_amount', 'customs_fee',
    'bank_commission', 'delivery_cost_rub'
]

df_model = df[feature_cols].fillna(0)
X = df_model.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Признаков: {len(feature_cols)}")
print(f"Записей:   {len(X_scaled)}")


# ============================================================
# ШАГ 3. ОБНАРУЖЕНИЕ АНОМАЛИЙ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Обнаружение аномалий")
print("=" * 60)

# 3.1 Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
labels_iso = iso_forest.fit_predict(X_scaled)
n_anomalies_iso = (labels_iso == -1).sum()
print(f"Isolation Forest:  найдено {n_anomalies_iso} аномалий ({n_anomalies_iso/len(X)*100:.2f}%)")

# 3.2 One-Class SVM
ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
labels_ocsvm = ocsvm.fit_predict(X_scaled)
n_anomalies_ocsvm = (labels_ocsvm == -1).sum()
print(f"One-Class SVM:     найдено {n_anomalies_ocsvm} аномалий ({n_anomalies_ocsvm/len(X)*100:.2f}%)")

# 3.3 Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labels_lof = lof.fit_predict(X_scaled)
n_anomalies_lof = (labels_lof == -1).sum()
print(f"Local Outlier Factor: найдено {n_anomalies_lof} аномалий ({n_anomalies_lof/len(X)*100:.2f}%)")

# Общие аномалии (найденные всеми тремя методами)
labels_all = labels_iso + labels_ocsvm + labels_lof
consensus = (labels_all <= -2).astype(int)  # минимум 2 из 3 согласны
n_consensus = consensus.sum()
print(f"\nСогласованные аномалии (>=2 из 3): {n_consensus} ({n_consensus/len(X)*100:.2f}%)")

df['anomaly_iso'] = (labels_iso == -1).astype(int)
df['anomaly_ocsvm'] = (labels_ocsvm == -1).astype(int)
df['anomaly_lof'] = (labels_lof == -1).astype(int)
df['anomaly_consensus'] = consensus


# ============================================================
# ШАГ 4. АНАЛИЗ АНОМАЛИЙ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Анализ аномалий")
print("=" * 60)

anomalies = df[df['anomaly_consensus'] == 1].copy()
print(f"Согласованных аномалий: {len(anomalies)}")

if len(anomalies) > 0:
    print("\nТОП-10 аномалий по сумме:")
    top_anom = anomalies.nlargest(10, 'total_amount')[
        ['date', 'counterparty', 'product_model', 'total_amount',
         'income_article', 'expense_article']
    ]
    print(top_anom.to_string(index=False))

    print("\nРаспределение аномалий по статьям:")
    print(anomalies['expense_article'].value_counts().head(10))
    print(anomalies['income_article'].value_counts().head(10))

    # Сохранение аномалий
    anomalies.to_csv(f'{REPORTS_DIR}/anomalies.csv', index=False)
    print(f"\nСохранён: {REPORTS_DIR}/anomalies.csv")


# ============================================================
# ШАГ 5. ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. Построение графиков")
print("=" * 60)

# PCA для 2D-визуализации
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 24. Аномалии в пространстве PCA (все три метода)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (labels, title) in zip(axes, [
    (labels_iso, 'Isolation Forest'),
    (labels_ocsvm, 'One-Class SVM'),
    (labels_lof, 'Local Outlier Factor')
]):
    normal_mask = labels == 1
    anomaly_mask = labels == -1
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
               c='steelblue', s=15, alpha=0.5, label='Норма')
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
               c='red', s=40, alpha=0.8, label='Аномалия', edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/24_anomaly_methods.png', dpi=150)
plt.close()
print("Сохранён: 24_anomaly_methods.png")

# 25. Согласованные аномалии
plt.figure(figsize=(12, 8))
normal_mask = df['anomaly_consensus'] == 0
anomaly_mask = df['anomaly_consensus'] == 1

plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
            c='steelblue', s=15, alpha=0.5, label='Норма')
plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
            c='red', s=80, alpha=0.9, label='Согласованная аномалия',
            edgecolors='black', linewidth=1.5)
plt.title(f'Согласованные аномалии (найдены ≥2 методами): {n_consensus}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/25_anomaly_consensus.png', dpi=150)
plt.close()
print("Сохранён: 25_anomaly_consensus.png")

# 26. Сравнение методов
methods = ['Isolation Forest', 'One-Class SVM', 'LOF', 'Согласованные']
counts = [n_anomalies_iso, n_anomalies_ocsvm, n_anomalies_lof, n_consensus]

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, counts, color=['steelblue', 'coral', 'teal', 'red'])
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(count), ha='center', fontsize=12, fontweight='bold')
plt.title('Количество обнаруженных аномалий разными методами')
plt.ylabel('Количество аномалий')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/26_anomaly_counts.png', dpi=150)
plt.close()
print("Сохранён: 26_anomaly_counts.png")

# 27. Распределение сумм: норма vs аномалии
plt.figure(figsize=(12, 6))
plt.hist(df[df['anomaly_consensus'] == 0]['total_amount'],
         bins=50, alpha=0.6, label='Норма', color='steelblue', log=True)
plt.hist(df[df['anomaly_consensus'] == 1]['total_amount'],
         bins=20, alpha=0.8, label='Аномалии', color='red', log=True)
plt.title('Распределение сумм транзакций: норма vs аномалии (лог. шкала)')
plt.xlabel('Сумма транзакции, руб.')
plt.ylabel('Частота (log)')
plt.legend()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/27_anomaly_amounts.png', dpi=150)
plt.close()
print("Сохранён: 27_anomaly_amounts.png")


# ============================================================
# ШАГ 6. ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

with open(f'{REPORTS_DIR}/anomaly_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: ОБНАРУЖЕНИЕ АНОМАЛИЙ\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Всего транзакций: {len(df)}\n")
    f.write(f"Признаков: {len(feature_cols)}\n\n")

    f.write("Результаты методов:\n")
    f.write(f"  Isolation Forest:      {n_anomalies_iso} ({n_anomalies_iso/len(X)*100:.2f}%)\n")
    f.write(f"  One-Class SVM:         {n_anomalies_ocsvm} ({n_anomalies_ocsvm/len(X)*100:.2f}%)\n")
    f.write(f"  Local Outlier Factor:  {n_anomalies_lof} ({n_anomalies_lof/len(X)*100:.2f}%)\n")
    f.write(f"  Согласованные (≥2):    {n_consensus} ({n_consensus/len(X)*100:.2f}%)\n\n")

    if len(anomalies) > 0:
        f.write("ТОП-10 аномалий по сумме:\n")
        f.write(top_anom.to_string(index=False))
        f.write("\n\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/anomaly_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Обнаружение аномалий завершено.")
print("=" * 60)
print(f"Isolation Forest:      {n_anomalies_iso} аномалий")
print(f"One-Class SVM:         {n_anomalies_ocsvm} аномалий")
print(f"Local Outlier Factor:  {n_anomalies_lof} аномалий")
print(f"Согласованные (≥2):    {n_consensus} аномалий")