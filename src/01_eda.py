"""
01_eda.py — Разведочный анализ данных (EDA) с проверкой качества.

Проект: Комплексный анализ, прогнозирование и оптимизация
        финансовых показателей компании на основе методов ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

DATA_PATH = 'data/robot_wash_extended.csv'
CLEAN_PATH = 'data/robot_wash_clean.csv'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Очистка старых графиков и отчётов (чтобы не плодились)
for f in glob.glob(f'{IMAGES_DIR}/*.png'):
    os.remove(f)
for f in glob.glob(f'{REPORTS_DIR}/*.txt'):
    os.remove(f)


# ============================================================
# ФУНКЦИЯ ПРОВЕРКИ КАЧЕСТВА ДАННЫХ
# ============================================================
def check_quality(df, stage="ДО очистки"):
    """Проверяет качество данных, возвращает отчёт в виде словаря."""
    print(f"\n--- Проверка качества: {stage} ---")

    report = {}
    report['Строк всего'] = len(df)
    report['Колонок'] = len(df.columns)

    # 1. Пропуски
    missing = df.isna().sum()
    missing = missing[missing > 0]
    report['Колонок с пропусками'] = len(missing)

    # 2. Дубликаты
    report['Дубликаты'] = int(df.duplicated().sum())

    # 3. Неверные даты (ИСПРАВЛЕНО: явный ISO-формат)
    if 'date' in df.columns:
        bad_dates = pd.to_datetime(
            df['date'], format='%Y-%m-%d', errors='coerce'
        ).isna().sum()
        report['Неверные даты'] = int(bad_dates)

    # 4. Отрицательные количества
    if 'quantity' in df.columns:
        neg_qty = (pd.to_numeric(df['quantity'], errors='coerce') < 0).sum()
        report['Отрицательные quantity'] = int(neg_qty)

    # 5. Текст вместо чисел
    text_in_num = 0
    for col in ['income', 'expense', 'exchange_rate']:
        if col in df.columns:
            as_num = pd.to_numeric(df[col], errors='coerce')
            bad = (as_num.isna() & df[col].notna()).sum()
            if bad > 0:
                report[f'Текст в {col}'] = int(bad)
                text_in_num += bad
    if text_in_num == 0:
        report['Текст в числовых колонках'] = 0

    # 6. Некорректные ТН ВЭД
    if 'tnved_code' in df.columns:
        def is_valid_tnved(code):
            code = str(code).strip().replace('.0', '')
            if code in ('', 'nan', 'None'):
                return True
            return code.isdigit() and len(code) == 10 and code != '9999999999'
        bad_tnved = sum(not is_valid_tnved(x) for x in df['tnved_code'])
        report['Некорректные ТН ВЭД'] = int(bad_tnved)

    # Вывод
    for k, v in report.items():
        print(f"  {k}: {v}")

    return report


def save_quality_report(report_before, report_after, path):
    """Сохраняет отчёт о качестве данных в текстовый файл."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ\n")
        f.write("=" * 50 + "\n\n")
        f.write("ДО ОЧИСТКИ:\n")
        for k, v in report_before.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nПОСЛЕ ОЧИСТКИ:\n")
        for k, v in report_after.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nОтчёт о качестве сохранён: {path}")


# ============================================================
# ШАГ 1. ЗАГРУЗКА ДАННЫХ
# ============================================================
print("=" * 60)
print("ШАГ 1. Загрузка данных")
print("=" * 60)

# Читаем ТН ВЭД как строку
df_raw = pd.read_csv(DATA_PATH, dtype={'tnved_code': str})
print(f"Строк загружено: {len(df_raw)}")
print(f"Колонок: {len(df_raw.columns)}")


# ============================================================
# ШАГ 2. ПРОВЕРКА КАЧЕСТВА ДО ОЧИСТКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Проверка качества ДО очистки")
print("=" * 60)

report_before = check_quality(df_raw, "ДО очистки")


# ============================================================
# ШАГ 3. ОЧИСТКА ДАННЫХ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Очистка данных")
print("=" * 60)

df = df_raw.copy()

# 3.1 Дата (ИСПРАВЛЕНО: явный ISO-формат)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
bad_dates = df['date'].isna().sum()
print(f"Неверных дат найдено: {bad_dates}")

# 3.2 Числовые колонки
numeric_cols = ['year', 'month', 'quantity', 'income', 'expense',
                'exchange_rate', 'customs_duty_rate', 'customs_duty_amount',
                'vat_rate', 'vat_amount', 'customs_fee', 'bank_commission',
                'delivery_cost_rub']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3.3 Отрицательные количества → модуль
neg_qty = (df['quantity'] < 0).sum()
print(f"Отрицательных quantity найдено: {neg_qty}")
df.loc[df['quantity'] < 0, 'quantity'] = df['quantity'].abs()

# 3.4 Заполнение пропусков
text_cols = ['legal_entity', 'counterparty', 'product_model',
             'income_article', 'income_group', 'expense_article',
             'expense_group', 'tnved_code', 'notes']
for col in text_cols:
    df[col] = df[col].fillna('')

for col in ['income', 'expense', 'customs_duty_amount', 'vat_amount',
            'customs_fee', 'bank_commission', 'delivery_cost_rub', 'quantity']:
    df[col] = df[col].fillna(0)

df['exchange_rate'] = df['exchange_rate'].fillna(df['exchange_rate'].median())

# 3.5 Удаление строк с некорректной датой
df = df.dropna(subset=['date'])
print(f"Строк удалено (неверная дата): {len(df_raw) - len(df)}")

# 3.6 Восстановление year/month/quarter из date
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = 'Q' + ((df['month'] - 1) // 3 + 1).astype(str)

# 3.7 Очистка ТН ВЭД
if 'tnved_code' in df.columns:
    df['tnved_code'] = (
        df['tnved_code'].astype(str)
        .str.replace('.0', '', regex=False)
        .replace('nan', '')
        .replace('9999999999', '')
        .str.strip()
    )

# 3.8 Удаление дубликатов
before_dedup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Удалено дубликатов: {before_dedup - len(df)}")

print(f"Строк после очистки: {len(df)}")

df.to_csv(CLEAN_PATH, index=False)
print(f"Очищенный датасет сохранён: {CLEAN_PATH}")


# ============================================================
# ШАГ 4. ПРОВЕРКА КАЧЕСТВА ПОСЛЕ ОЧИСТКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Проверка качества ПОСЛЕ очистки")
print("=" * 60)

report_after = check_quality(df, "ПОСЛЕ очистки")

save_quality_report(report_before, report_after, f'{REPORTS_DIR}/data_quality.txt')


# ============================================================
# ШАГ 5. ОБЩИЕ СТАТИСТИКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. Общие статистики")
print("=" * 60)

total_income = df['income'].sum()
total_expense = df['expense'].sum()
profit = total_income - total_expense

print(f"Общая выручка:      {total_income:>15,.0f} руб.")
print(f"Общие расходы:      {total_expense:>15,.0f} руб.")
print(f"Прибыль:            {profit:>15,.0f} руб.")
print(f"Рентабельность:     {profit / total_income * 100:>15.2f} %")

with open(f'{REPORTS_DIR}/eda_summary.txt', 'w', encoding='utf-8') as f:
    f.write("ОБЩИЕ ПОКАЗАТЕЛИ\n")
    f.write("=" * 40 + "\n")
    f.write(f"Всего строк:        {len(df)}\n")
    f.write(f"Период:             {df['date'].min().date()} - {df['date'].max().date()}\n")
    f.write(f"Общая выручка:      {total_income:,.0f} руб.\n")
    f.write(f"Общие расходы:      {total_expense:,.0f} руб.\n")
    f.write(f"Прибыль:            {profit:,.0f} руб.\n")
    f.write(f"Рентабельность:     {profit / total_income * 100:.2f} %\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/eda_summary.txt")


# ============================================================
# ШАГ 6. ВИЗУАЛИЗАЦИЯ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 6. Построение графиков")
print("=" * 60)

df['year_month'] = df['date'].dt.to_period('M').astype(str)

# 6.1 Динамика выручки, расходов и прибыли
monthly = df.groupby('year_month').agg(
    income=('income', 'sum'),
    expense=('expense', 'sum')
).reset_index()
monthly['profit'] = monthly['income'] - monthly['expense']

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(monthly['year_month'], monthly['income'], marker='o', label='Выручка', color='green')
ax.plot(monthly['year_month'], monthly['expense'], marker='s', label='Расходы', color='red')
ax.plot(monthly['year_month'], monthly['profit'], marker='^', label='Прибыль', color='blue')
ax.set_title('Динамика выручки, расходов и прибыли по месяцам')
ax.set_xlabel('Месяц')
ax.set_ylabel('Сумма, руб.')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/01_monthly_dynamics.png', dpi=150)
plt.close()
print("Сохранён: 01_monthly_dynamics.png")

# 6.2 Структура расходов
expense_groups = df[df['expense'] > 0].groupby('expense_group')['expense'].sum()
expense_groups = expense_groups[expense_groups.index != '']

fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(expense_groups.values, labels=expense_groups.index, autopct='%1.1f%%',
       startangle=90, colors=sns.color_palette('Reds', len(expense_groups)))
ax.set_title('Структура расходов по группам')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/02_expense_structure.png', dpi=150)
plt.close()
print("Сохранён: 02_expense_structure.png")

# 6.3 Структура доходов
income_articles = df[df['income'] > 0].groupby('income_article')['income'].sum()
income_articles = income_articles[income_articles.index != '']

fig, ax = plt.subplots(figsize=(10, 8))
ax.pie(income_articles.values, labels=income_articles.index, autopct='%1.1f%%',
       startangle=90, colors=sns.color_palette('Greens', len(income_articles)))
ax.set_title('Структура доходов по статьям')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/03_income_structure.png', dpi=150)
plt.close()
print("Сохранён: 03_income_structure.png")

# 6.4 Продажи по продуктам
sales = df[df['income_article'] == 'Продажа робота'].groupby('product_model').agg(
    total=('income', 'sum'),
    count=('quantity', 'sum')
).reset_index()

if len(sales) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sales['product_model'], sales['total'], color='skyblue')
    for i, v in enumerate(sales['total']):
        ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom')
    ax.set_title('Выручка по моделям роботов')
    ax.set_xlabel('Модель')
    ax.set_ylabel('Выручка, руб.')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/04_product_sales.png', dpi=150)
    plt.close()
    print("Сохранён: 04_product_sales.png")

# 6.5 Топ-10 контрагентов
top_clients = df[df['income'] > 0].groupby('counterparty')['income'].sum() \
    .sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
top_clients.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Топ-10 контрагентов по выручке')
ax.set_xlabel('Выручка, руб.')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/05_top_clients.png', dpi=150)
plt.close()
print("Сохранён: 05_top_clients.png")

# 6.6 Влияние курса валют
purchases = df[(df['expense_article'] == 'Закупка товара') & (df['expense'] > 0)]

if len(purchases) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(purchases['exchange_rate'], purchases['expense'],
               alpha=0.6, color='darkred', s=60)
    ax.set_title('Зависимость суммы закупки от курса валюты')
    ax.set_xlabel('Курс валюты, руб.')
    ax.set_ylabel('Сумма закупки, руб.')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/06_exchange_rate.png', dpi=150)
    plt.close()
    print("Сохранён: 06_exchange_rate.png")

# 6.7 Продажи по кварталам
robot_sales = df[df['income_article'] == 'Продажа робота'].copy()
if len(robot_sales) > 0:
    quarterly = robot_sales.groupby(['year', 'quarter']).agg(
        quantity=('quantity', 'sum'),
        revenue=('income', 'sum')
    ).reset_index()
    quarterly['label'] = quarterly['year'].astype(str) + '-' + quarterly['quarter']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(quarterly['label'], quarterly['quantity'], color='teal')
    ax.set_title('Количество проданных роботов по кварталам')
    ax.set_xlabel('Квартал')
    ax.set_ylabel('Количество, шт.')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/07_quarterly_sales.png', dpi=150)
    plt.close()
    print("Сохранён: 07_quarterly_sales.png")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! EDA завершён.")
print("=" * 60)
print(f"Очищенный датасет: {CLEAN_PATH}")
print(f"Отчёт о качестве:  {REPORTS_DIR}/data_quality.txt")
print(f"Сводка:            {REPORTS_DIR}/eda_summary.txt")
print(f"Графики:           {IMAGES_DIR}/01-07*.png")