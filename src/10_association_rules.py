"""
10_association_rules.py — Поиск ассоциативных правил (Apriori, FP-Growth).

Задача: найти часто встречающиеся комбинации товаров/услуг в одной сделке.

Читает:  data/robot_wash_features.csv
Пишет:   reports/association_rules_report.txt
         reports/association_rules.csv
         images/43-46*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

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
# ШАГ 2. ФОРМИРОВАНИЕ "КОРЗИН"
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Формирование транзакций (корзин)")
print("=" * 60)

# Группируем по дате + контрагенту — это одна сделка
df['txn'] = df['date'].astype(str) + '_' + df['counterparty'].astype(str)

transactions = []
for txn, group in df.groupby('txn'):
    items = set()
    for _, row in group.iterrows():
        if row['income_article']:
            items.add(f"IN:{row['income_article']}")
        if row['expense_article']:
            items.add(f"OUT:{row['expense_article']}")
        if row['product_model']:
            items.add(f"MODEL:{row['product_model']}")
    if len(items) >= 2:
        transactions.append(list(items))

print(f"Транзакций (сделок): {len(transactions)}")
print(f"Пример: {transactions[0] if transactions else 'нет'}")


# ============================================================
# ШАГ 3. ONE-HOT ENCODING
# ============================================================
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_te = pd.DataFrame(te_array, columns=te.columns_)
print(f"Матрица: {df_te.shape}")


# ============================================================
# ШАГ 4. APRIORI
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Apriori")
print("=" * 60)

frequent_apriori = apriori(df_te, min_support=0.02, use_colnames=True)
print(f"Частых наборов (Apriori): {len(frequent_apriori)}")

if len(frequent_apriori) > 0:
    rules_apriori = association_rules(frequent_apriori, metric="confidence", min_threshold=0.3)
    rules_apriori = rules_apriori.sort_values('lift', ascending=False)
    print(f"Правил (Apriori): {len(rules_apriori)}")
    print("\nТОП-10 правил по lift:")
    for _, r in rules_apriori.head(10).iterrows():
        print(f"  {set(r['antecedents'])} → {set(r['consequents'])} "
              f"(sup={r['support']:.3f}, conf={r['confidence']:.3f}, lift={r['lift']:.2f})")
else:
    rules_apriori = pd.DataFrame()


# ============================================================
# ШАГ 5. FP-GROWTH
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. FP-Growth")
print("=" * 60)

frequent_fp = fpgrowth(df_te, min_support=0.02, use_colnames=True)
print(f"Частых наборов (FP-Growth): {len(frequent_fp)}")

if len(frequent_fp) > 0:
    rules_fp = association_rules(frequent_fp, metric="confidence", min_threshold=0.3)
    rules_fp = rules_fp.sort_values('lift', ascending=False)
    print(f"Правил (FP-Growth): {len(rules_fp)}")
else:
    rules_fp = pd.DataFrame()


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 43. Топ-15 частых наборов
if len(frequent_apriori) > 0:
    top_freq = frequent_apriori.nlargest(15, 'support').copy()
    top_freq['itemset'] = top_freq['itemsets'].apply(lambda x: ', '.join(list(x)))

    plt.figure(figsize=(12, 7))
    plt.barh(top_freq['itemset'], top_freq['support'], color='steelblue')
    plt.title('ТОП-15 частых наборов (Apriori, min_support=0.02)')
    plt.xlabel('Поддержка (support)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/43_assoc_frequent.png', dpi=150)
    plt.close()
    print("Сохранён: 43_assoc_frequent.png")

# 44. Support vs Confidence (scatter)
if len(rules_apriori) > 0:
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(rules_apriori['support'], rules_apriori['confidence'],
                          c=rules_apriori['lift'], cmap='viridis',
                          s=60, alpha=0.7, edgecolors='black')
    plt.colorbar(scatter, label='Lift')
    plt.title('Правила: Support vs Confidence (цвет = Lift)')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/44_assoc_scatter.png', dpi=150)
    plt.close()
    print("Сохранён: 44_assoc_scatter.png")

# 45. Топ-15 правил по lift
if len(rules_apriori) > 0:
    top_rules = rules_apriori.head(15).copy()
    top_rules['label'] = top_rules.apply(
        lambda r: f"{', '.join(list(r['antecedents']))} → {', '.join(list(r['consequents']))}",
        axis=1
    )

    plt.figure(figsize=(12, 7))
    plt.barh(top_rules['label'], top_rules['lift'], color='teal')
    plt.title('ТОП-15 ассоциативных правил по Lift')
    plt.xlabel('Lift')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/45_assoc_top_rules.png', dpi=150)
    plt.close()
    print("Сохранён: 45_assoc_top_rules.png")

# 46. Сравнение Apriori vs FP-Growth
plt.figure(figsize=(10, 6))
methods = ['Apriori', 'FP-Growth']
counts = [len(frequent_apriori), len(frequent_fp)]
bars = plt.bar(methods, counts, color=['steelblue', 'coral'])
for bar, c in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(c), ha='center', fontsize=12, fontweight='bold')
plt.title('Сравнение Apriori и FP-Growth (количество частых наборов)')
plt.ylabel('Количество наборов')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/46_assoc_compare.png', dpi=150)
plt.close()
print("Сохранён: 46_assoc_compare.png")


# ============================================================
# ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

if len(rules_apriori) > 0:
    rules_apriori.to_csv(f'{REPORTS_DIR}/association_rules.csv', index=False)

with open(f'{REPORTS_DIR}/association_rules_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: АССОЦИАТИВНЫЕ ПРАВИЛА\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Транзакций: {len(transactions)}\n")
    f.write(f"Уникальных items: {len(te.columns_)}\n\n")

    f.write("APRIORI:\n")
    f.write(f"  Частых наборов: {len(frequent_apriori)}\n")
    f.write(f"  Правил: {len(rules_apriori)}\n\n")

    if len(rules_apriori) > 0:
        f.write("ТОП-15 правил по lift:\n")
        for _, r in rules_apriori.head(15).iterrows():
            f.write(f"  {set(r['antecedents'])} → {set(r['consequents'])} "
                    f"(sup={r['support']:.3f}, conf={r['confidence']:.3f}, lift={r['lift']:.2f})\n")

    f.write("\nFP-GROWTH:\n")
    f.write(f"  Частых наборов: {len(frequent_fp)}\n")
    f.write(f"  Правил: {len(rules_fp)}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/association_rules_report.txt")


print("\n" + "=" * 60)
print("ГОТОВО! Ассоциативные правила завершены.")
print("=" * 60)