"""
09_dimreduction.py — Снижение размерности.

Методы: PCA, t-SNE, LDA, TruncatedSVD, UMAP (опционально).

Читает:  data/robot_wash_features.csv
Пишет:   reports/dimreduction_report.txt
         images/38-43*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
print(f"Строк: {len(df)}, колонок: {len(df.columns)}")


# ============================================================
# ШАГ 2. ПОДГОТОВКА МАТРИЦЫ ПРИЗНАКОВ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Подготовка признаков")
print("=" * 60)

feature_cols = [
    'quantity', 'exchange_rate', 'customs_duty_rate',
    'customs_duty_amount', 'vat_rate', 'vat_amount',
    'customs_fee', 'bank_commission', 'delivery_cost_rub',
    'month', 'day_of_week', 'quarter_num',
    'customs_share', 'vat_share', 'logistics_share', 'bank_share'
]

df_model = df[feature_cols + ['is_income', 'income_article']].fillna(0).copy()
X = df_model[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Матрица: {X_scaled.shape}")


# ============================================================
# ШАГ 3. PCA
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. PCA")
print("=" * 60)

pca = PCA(n_components=min(len(feature_cols), 10), random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Объяснённая дисперсия по компонентам:")
cumsum = 0
for i, v in enumerate(pca.explained_variance_ratio_):
    cumsum += v
    print(f"  PC{i+1}: {v*100:6.2f}% (накопл. {cumsum*100:6.2f}%)")


# ============================================================
# ШАГ 4. t-SNE
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. t-SNE")
print("=" * 60)

# t-SNE на подвыборке (быстро)
sample_size = min(2000, len(X_scaled))
idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx]
y_sample = df_model['is_income'].values[idx]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_sample)
print(f"t-SNE: {X_tsne.shape}")


# ============================================================
# ШАГ 5. LDA
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. LDA (линейный дискриминантный анализ)")
print("=" * 60)

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_scaled, df_model['is_income'].values)
print(f"LDA: {X_lda.shape}")
print(f"Объяснённая дисперсия: {lda.explained_variance_ratio_.sum()*100:.2f}%")


# ============================================================
# ШАГ 6. TruncatedSVD
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 6. TruncatedSVD")
print("=" * 60)

svd = TruncatedSVD(n_components=5, random_state=42)
X_svd = svd.fit_transform(X_scaled)
print(f"SVD: {X_svd.shape}")
print(f"Объяснённая дисперсия: {svd.explained_variance_ratio_.sum()*100:.2f}%")


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 38. Объяснённая дисперсия PCA
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100, color='steelblue', alpha=0.7, label='Отдельно')
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_) * 100,
         color='red', marker='o', label='Накопленная')
plt.title('PCA: объяснённая дисперсия по компонентам')
plt.xlabel('Компонента')
plt.ylabel('Дисперсия, %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/38_dimreduction_pca_var.png', dpi=150)
plt.close()
print("Сохранён: 38_dimreduction_pca_var.png")

# 39. PCA 2D — цвет по is_income
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=df_model['is_income'], cmap='coolwarm',
                      s=15, alpha=0.6)
plt.colorbar(scatter, label='is_income (0=расход, 1=доход)')
plt.title('PCA: 2D-проекция транзакций')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/39_dimreduction_pca_2d.png', dpi=150)
plt.close()
print("Сохранён: 39_dimreduction_pca_2d.png")

# 40. t-SNE
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=y_sample, cmap='viridis', s=20, alpha=0.7)
plt.colorbar(scatter, label='is_income')
plt.title(f't-SNE (выборка {sample_size} транзакций)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/40_dimreduction_tsne.png', dpi=150)
plt.close()
print("Сохранён: 40_dimreduction_tsne.png")

# 41. LDA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_lda[df_model['is_income'].values == 0], bins=40,
         alpha=0.6, color='red', label='Расход')
plt.hist(X_lda[df_model['is_income'].values == 1], bins=40,
         alpha=0.6, color='green', label='Доход')
plt.title('LDA: распределение по классам')
plt.xlabel('LD1')
plt.ylabel('Частота')
plt.legend()

# 42. Вклад признаков в PC1/PC2
plt.subplot(1, 2, 2)
loadings = pd.DataFrame(pca.components_[:2].T,
                        columns=['PC1', 'PC2'],
                        index=feature_cols)
loadings.plot(kind='barh', ax=plt.gca(), color=['steelblue', 'coral'])
plt.title('Вклад признаков в PC1 и PC2')
plt.xlabel('Загрузка')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/41_dimreduction_lda.png', dpi=150)
plt.close()
print("Сохранён: 41_dimreduction_lda.png")

# 43. SVD 2D
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1],
                      c=df_model['is_income'], cmap='plasma',
                      s=15, alpha=0.6)
plt.colorbar(scatter, label='is_income')
plt.title('TruncatedSVD: 2D-проекция')
plt.xlabel('SVD1')
plt.ylabel('SVD2')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/42_dimreduction_svd.png', dpi=150)
plt.close()
print("Сохранён: 42_dimreduction_svd.png")


# ============================================================
# ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

with open(f'{REPORTS_DIR}/dimreduction_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: СНИЖЕНИЕ РАЗМЕРНОСТИ\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Исходная матрица: {X_scaled.shape}\n")
    f.write(f"Признаков: {len(feature_cols)}\n\n")

    f.write("PCA:\n")
    f.write(f"  Компонент: {X_pca.shape[1]}\n")
    f.write(f"  Объяснено дисперсии: {pca.explained_variance_ratio_.sum()*100:.2f}%\n")
    for i, v in enumerate(pca.explained_variance_ratio_):
        f.write(f"    PC{i+1}: {v*100:.2f}%\n")
    f.write("\n")

    f.write("t-SNE:\n")
    f.write(f"  Выборка: {sample_size}\n")
    f.write(f"  Компонент: 2\n\n")

    f.write("LDA:\n")
    f.write(f"  Компонент: 1\n")
    f.write(f"  Объяснено: {lda.explained_variance_ratio_.sum()*100:.2f}%\n\n")

    f.write("TruncatedSVD:\n")
    f.write(f"  Компонент: 5\n")
    f.write(f"  Объяснено: {svd.explained_variance_ratio_.sum()*100:.2f}%\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/dimreduction_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Снижение размерности завершено.")
print("=" * 60)
print(f"PCA:          {X_pca.shape[1]} компонент, {pca.explained_variance_ratio_.sum()*100:.2f}% дисперсии")
print(f"t-SNE:        2 компоненты (визуализация)")
print(f"LDA:          1 компонента, {lda.explained_variance_ratio_.sum()*100:.2f}% дисперсии")
print(f"TruncatedSVD: 5 компонент, {svd.explained_variance_ratio_.sum()*100:.2f}% дисперсии")