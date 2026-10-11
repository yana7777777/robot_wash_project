"""
05_clustering.py — Кластеризация клиентов и транзакций.

Читает:  data/robot_wash_features.csv
Пишет:   reports/clustering_report.txt
         images/18-23*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

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


# ============================================================
# ЗАДАЧА 1. КЛАСТЕРИЗАЦИЯ КЛИЕНТОВ
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 1. Кластеризация клиентов (контрагентов)")
print("=" * 70)

# Агрегируем по контрагенту
clients = df[df['counterparty'] != ''].groupby('counterparty').agg(
    total_income=('income', 'sum'),
    total_expense=('expense', 'sum'),
    transactions=('income', 'count'),
    avg_income=('income', 'mean'),
    avg_expense=('expense', 'mean'),
    total_quantity=('quantity', 'sum'),
).reset_index()

clients['total_amount'] = clients['total_income'] + clients['total_expense']
clients = clients[clients['transactions'] >= 2]
print(f"Контрагентов: {len(clients)}")

# Признаки для кластеризации
client_features = ['total_income', 'total_expense', 'transactions',
                   'avg_income', 'avg_expense', 'total_quantity']
X_clients = clients[client_features].values

# Масштабирование
scaler_c = StandardScaler()
X_clients_scaled = scaler_c.fit_transform(X_clients)

# Определение оптимального K методом локтя
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clients_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, marker='o', color='steelblue', linewidth=2)
plt.title('Метод локтя: выбор оптимального числа кластеров (клиенты)')
plt.xlabel('Число кластеров (K)')
plt.ylabel('Инерция')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/18_clustering_elbow.png', dpi=150)
plt.close()
print("Сохранён: 18_clustering_elbow.png")

# Сравнение методов кластеризации при K=4
K_OPT = 4

clustering_methods = {
    'KMeans': KMeans(n_clusters=K_OPT, random_state=42, n_init=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=K_OPT),
    'GaussianMixture': GaussianMixture(n_components=K_OPT, random_state=42),
}

results_clustering = []
cluster_labels = {}

for name, model in clustering_methods.items():
    if name == 'GaussianMixture':
        labels = model.fit_predict(X_clients_scaled)
    else:
        labels = model.fit_predict(X_clients_scaled)

    sil = silhouette_score(X_clients_scaled, labels)
    db = davies_bouldin_score(X_clients_scaled, labels)
    ch = calinski_harabasz_score(X_clients_scaled, labels)

    results_clustering.append({
        'Метод': name, 'Silhouette': sil,
        'Davies-Bouldin': db, 'Calinski-Harabasz': ch
    })
    cluster_labels[name] = labels

    print(f"{name:18s} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:,.0f}")

# DBSCAN отдельно (без указания числа кластеров)
dbscan = DBSCAN(eps=1.5, min_samples=3)
labels_db = dbscan.fit_predict(X_clients_scaled)
n_db_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)

if n_db_clusters >= 2:
    sil_db = silhouette_score(X_clients_scaled, labels_db)
    db_db = davies_bouldin_score(X_clients_scaled, labels_db)
    ch_db = calinski_harabasz_score(X_clients_scaled, labels_db)

    results_clustering.append({
        'Метод': 'DBSCAN', 'Silhouette': sil_db,
        'Davies-Bouldin': db_db, 'Calinski-Harabasz': ch_db
    })
    cluster_labels['DBSCAN'] = labels_db

    print(f"{'DBSCAN':18s} | Silhouette: {sil_db:.4f} | DB: {db_db:.4f} | CH: {ch_db:,.0f} | Кластеров: {n_db_clusters}")
else:
    print(f"{'DBSCAN':18s} | Не нашёл достаточно кластеров")

results_cl_df = pd.DataFrame(results_clustering).sort_values('Silhouette', ascending=False)
best_method = results_cl_df.iloc[0]['Метод']
print(f"\nЛучший метод: {best_method}")

# Добавим метки лучшего метода к клиентам
clients['cluster'] = cluster_labels[best_method]

# Сохраняем
clients.to_csv('data/clients_clusters.csv', index=False)
print("Сохранён: data/clients_clusters.csv")

# ============================================================
# ГРАФИКИ
# ============================================================

# 19. Сравнение методов кластеризации
plt.figure(figsize=(12, 6))
plt.barh(results_cl_df['Метод'], results_cl_df['Silhouette'], color='mediumseagreen')
plt.title('Сравнение методов кластеризации (Silhouette, чем больше — тем лучше)')
plt.xlabel('Silhouette')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/19_clustering_methods.png', dpi=150)
plt.close()
print("Сохранён: 19_clustering_methods.png")

# 20. PCA-визуализация кластеров клиентов
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clients_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=clients['cluster'], cmap='viridis',
                      s=60, alpha=0.7, edgecolors='black')
plt.colorbar(scatter, label='Кластер')
plt.title(f'Кластеры клиентов в пространстве PCA ({best_method})')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/20_clustering_pca.png', dpi=150)
plt.close()
print("Сохранён: 20_clustering_pca.png")

# 21. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=min(15, len(clients)-1))
X_tsne = tsne.fit_transform(X_clients_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=clients['cluster'], cmap='plasma',
                      s=60, alpha=0.7, edgecolors='black')
plt.colorbar(scatter, label='Кластер')
plt.title(f'Кластеры клиентов в пространстве t-SNE ({best_method})')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/21_clustering_tsne.png', dpi=150)
plt.close()
print("Сохранён: 21_clustering_tsne.png")

# 22. Профили кластеров
cluster_profile = clients.groupby('cluster')[client_features].mean()

fig, ax = plt.subplots(figsize=(12, 6))
cluster_profile.T.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Профили кластеров клиентов')
ax.set_xlabel('Признак')
ax.set_ylabel('Среднее значение')
ax.legend(title='Кластер')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/22_clustering_profiles.png', dpi=150)
plt.close()
print("Сохранён: 22_clustering_profiles.png")


# ============================================================
# ЗАДАЧА 2. КЛАСТЕРИЗАЦИЯ ТРАНЗАКЦИЙ
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 2. Кластеризация транзакций")
print("=" * 70)

# Только транзакции с ненулевой суммой
df_trans = df[df['total_amount'] > 0].copy()
trans_features = ['total_amount', 'exchange_rate', 'customs_share',
                  'vat_share', 'logistics_share', 'quantity']
df_trans = df_trans[trans_features].dropna()

print(f"Транзакций: {len(df_trans)}")

scaler_t = StandardScaler()
X_trans_scaled = scaler_t.fit_transform(df_trans.values)

# Кластеризация K-Means
km_trans = KMeans(n_clusters=4, random_state=42, n_init=10)
trans_labels = km_trans.fit_predict(X_trans_scaled)

sil_t = silhouette_score(X_trans_scaled, trans_labels)
db_t = davies_bouldin_score(X_trans_scaled, trans_labels)
ch_t = calinski_harabasz_score(X_trans_scaled, trans_labels)

print(f"KMeans: Silhouette = {sil_t:.4f}, DB = {db_t:.4f}, CH = {ch_t:,.0f}")

# PCA-визуализация транзакций
pca_t = PCA(n_components=2, random_state=42)
X_trans_pca = pca_t.fit_transform(X_trans_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_trans_pca[:, 0], X_trans_pca[:, 1],
                      c=trans_labels, cmap='coolwarm',
                      s=20, alpha=0.6)
plt.colorbar(scatter, label='Кластер')
plt.title(f'Кластеры транзакций в пространстве PCA (Silhouette = {sil_t:.3f})')
plt.xlabel(f'PC1 ({pca_t.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_t.explained_variance_ratio_[1]*100:.1f}%)')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/23_clustering_transactions.png', dpi=150)
plt.close()
print("Сохранён: 23_clustering_transactions.png")


# ============================================================
# ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

with open(f'{REPORTS_DIR}/clustering_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: КЛАСТЕРИЗАЦИЯ\n")
    f.write("=" * 70 + "\n\n")

    f.write("ЗАДАЧА 1. Кластеризация клиентов\n")
    f.write("-" * 70 + "\n")
    f.write(f"Всего контрагентов (с >=2 сделками): {len(clients)}\n")
    f.write(f"Признаки: {', '.join(client_features)}\n\n")
    f.write("Сравнение методов кластеризации:\n")
    f.write(results_cl_df.to_string(index=False))
    f.write(f"\n\nЛучший метод: {best_method}\n")
    f.write(f"Silhouette: {results_cl_df.iloc[0]['Silhouette']:.4f}\n\n")

    f.write("Состав кластеров:\n")
    for cl in sorted(clients['cluster'].unique()):
        count = (clients['cluster'] == cl).sum()
        f.write(f"  Кластер {cl}: {count} контрагентов\n")

    f.write("\n\nЗАДАЧА 2. Кластеризация транзакций\n")
    f.write("-" * 70 + "\n")
    f.write(f"Всего транзакций: {len(df_trans)}\n")
    f.write(f"KMeans (K=4): Silhouette = {sil_t:.4f}, DB = {db_t:.4f}, CH = {ch_t:,.0f}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/clustering_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Кластеризация завершена.")
print("=" * 60)
print(f"Лучший метод: {best_method}")
print(f"Silhouette:   {results_cl_df.iloc[0]['Silhouette']:.4f}")