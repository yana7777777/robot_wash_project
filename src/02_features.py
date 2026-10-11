"""
02_features.py — Создание признаков (Feature Engineering).

Читает:  data/robot_wash_clean.csv
Пишет:   data/robot_wash_features.csv  (транзакции)
         data/robot_wash_monthly.csv   (агрегация по месяцам)
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

CLEAN_PATH = 'data/robot_wash_clean.csv'
FEATURES_PATH = 'data/robot_wash_features.csv'
MONTHLY_PATH = 'data/robot_wash_monthly.csv'
REPORTS_DIR = 'reports'

os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# ШАГ 1. ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("ШАГ 1. Загрузка очищенного датасета")
print("=" * 60)

df = pd.read_csv(CLEAN_PATH, dtype={'tnved_code': str}, parse_dates=['date'])
print(f"Строк: {len(df)}")
print(f"Колонок: {len(df.columns)}")


# ============================================================
# ШАГ 2. БАЗОВЫЕ ПРИЗНАКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 2. Создание базовых признаков")
print("=" * 60)

# 2.1 Прибыль по строке (для продаж — доход, для расходов — 0)
df['profit'] = df['income'] - df['expense']

# 2.2 Общая сумма операции
df['total_amount'] = df['income'] + df['expense']

# 2.3 Признак типа операции
df['is_income'] = (df['income'] > 0).astype(int)
df['is_expense'] = (df['expense'] > 0).astype(int)

# 2.4 Флаг импорта (закупка у Китая)
df['is_import'] = ((df['expense_article'] == 'Закупка товара') &
                   (df['counterparty'] == 'Leisuwash China')).astype(int)

# 2.5 Флаг продажи робота
df['is_robot_sale'] = (df['income_article'] == 'Продажа робота').astype(int)

# 2.6 Флаг сервисных услуг
df['is_service'] = (df['income_article'].isin(
    ['Монтаж', 'Сервисное обслуживание', 'Продажа запчастей'])).astype(int)

# 2.7 Флаг таможенных платежей
df['is_customs'] = (df['expense_group'] == 'Таможня').astype(int)

# 2.8 Флаг логистики
df['is_logistics'] = (df['expense_group'] == 'Логистика').astype(int)

print("Созданы флаги: is_income, is_expense, is_import, is_robot_sale, is_service, is_customs, is_logistics")


# ============================================================
# ШАГ 3. ВРЕМЕННЫЕ ПРИЗНАКИ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 3. Временные признаки")
print("=" * 60)

df['day_of_week'] = df['date'].dt.dayofweek        # 0 = понедельник
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

# Сезон
def get_season(month):
    if month in [12, 1, 2]:
        return 'Зима'
    elif month in [3, 4, 5]:
        return 'Весна'
    elif month in [6, 7, 8]:
        return 'Лето'
    else:
        return 'Осень'

df['season'] = df['month'].apply(get_season)

# Квартал как число
df['quarter_num'] = df['quarter'].str.replace('Q', '').astype(int)

# Дней с начала периода (для трендов)
min_date = df['date'].min()
df['days_since_start'] = (df['date'] - min_date).dt.days

print("Созданы признаки: day_of_week, day_of_month, week_of_year, season, quarter_num, days_since_start")


# ============================================================
# ШАГ 4. ДОЛИ И СТРУКТУРА ЗАТРАТ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 4. Доли затрат")
print("=" * 60)

# 4.1 Доля таможенных платежей в расходах на импорт
df['customs_share'] = np.where(
    df['expense'] > 0,
    (df['customs_duty_amount'].fillna(0) +
     df['vat_amount'].fillna(0) +
     df['customs_fee'].fillna(0)) / df['expense'],
    0
)

# 4.2 Доля НДС в расходах
df['vat_share'] = np.where(
    df['expense'] > 0,
    df['vat_amount'].fillna(0) / df['expense'],
    0
)

# 4.3 Доля логистики в расходах
df['logistics_share'] = np.where(
    df['expense'] > 0,
    df['delivery_cost_rub'].fillna(0) / df['expense'],
    0
)

# 4.4 Доля банковской комиссии
df['bank_share'] = np.where(
    df['expense'] > 0,
    df['bank_commission'].fillna(0) / df['expense'],
    0
)

print("Созданы признаки: customs_share, vat_share, logistics_share, bank_share")


# ============================================================
# ШАГ 5. ПРИЗНАКИ ДЛЯ ПРОДАЖ (маржинальность)
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 5. Признаки для продаж")
print("=" * 60)

# 5.1 Цена за единицу (только для продаж)
df['unit_price'] = np.where(
    (df['quantity'] > 0) & (df['income'] > 0),
    df['income'] / df['quantity'],
    0
)

# 5.2 Закупочная цена за единицу (только для импорта)
df['unit_cost'] = np.where(
    (df['quantity'] > 0) & (df['expense_article'] == 'Закупка товара'),
    df['expense'] / df['quantity'],
    0
)

print("Созданы признаки: unit_price, unit_cost")


# ============================================================
# ШАГ 6. ИТОГОВЫЙ ДАТАСЕТ ПО ТРАНЗАКЦИЯМ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 6. Сохранение датасета транзакций")
print("=" * 60)

df.to_csv(FEATURES_PATH, index=False)
print(f"Сохранён: {FEATURES_PATH}")
print(f"Строк: {len(df)}, колонок: {len(df.columns)}")


# ============================================================
# ШАГ 7. АГРЕГАЦИЯ ПО МЕСЯЦАМ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 7. Агрегация по месяцам")
print("=" * 60)

df['year_month'] = df['date'].dt.to_period('M').astype(str)

monthly = df.groupby('year_month').agg(
    income=('income', 'sum'),
    expense=('expense', 'sum'),
    quantity_sold=('quantity', lambda x: x[df.loc[x.index, 'is_robot_sale'] == 1].sum()),
    robot_sales_count=('is_robot_sale', 'sum'),
    service_count=('is_service', 'sum'),
    import_count=('is_import', 'sum'),
    avg_exchange_rate=('exchange_rate', 'mean'),
    customs_total=('customs_duty_amount', 'sum'),
    vat_total=('vat_amount', 'sum'),
    logistics_total=('delivery_cost_rub', 'sum'),
    bank_commission_total=('bank_commission', 'sum'),
    transactions=('income', 'count'),
).reset_index()

monthly['profit'] = monthly['income'] - monthly['expense']
monthly['margin'] = np.where(
    monthly['income'] > 0,
    monthly['profit'] / monthly['income'] * 100,
    0
)
monthly['avg_check'] = np.where(
    monthly['robot_sales_count'] > 0,
    monthly['income'] / monthly['robot_sales_count'],
    0
)

# Временные признаки
monthly['year'] = monthly['year_month'].str[:4].astype(int)
monthly['month'] = monthly['year_month'].str[5:7].astype(int)
monthly['quarter'] = 'Q' + ((monthly['month'] - 1) // 3 + 1).astype(str)

# Лаги (для временных рядов)
monthly = monthly.sort_values('year_month').reset_index(drop=True)
monthly['profit_lag1'] = monthly['profit'].shift(1)
monthly['income_lag1'] = monthly['income'].shift(1)
monthly['income_lag3'] = monthly['income'].shift(3)
monthly['profit_rolling3'] = monthly['profit'].rolling(3).mean()

monthly.to_csv(MONTHLY_PATH, index=False)
print(f"Сохранён: {MONTHLY_PATH}")
print(f"Строк: {len(monthly)}, колонок: {len(monthly.columns)}")


# ============================================================
# ШАГ 8. ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("ШАГ 8. Отчёт о созданных признаках")
print("=" * 60)

new_features = [
    'profit', 'total_amount', 'is_income', 'is_expense', 'is_import',
    'is_robot_sale', 'is_service', 'is_customs', 'is_logistics',
    'day_of_week', 'day_of_month', 'week_of_year', 'season', 'quarter_num',
    'days_since_start', 'customs_share', 'vat_share', 'logistics_share',
    'bank_share', 'unit_price', 'unit_cost'
]

with open(f'{REPORTS_DIR}/features_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ О СОЗДАННЫХ ПРИЗНАКАХ\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Датасет транзакций: {FEATURES_PATH}\n")
    f.write(f"  Строк: {len(df)}\n")
    f.write(f"  Колонок: {len(df.columns)}\n\n")
    f.write(f"Датасет по месяцам: {MONTHLY_PATH}\n")
    f.write(f"  Строк: {len(monthly)}\n")
    f.write(f"  Колонок: {len(monthly.columns)}\n\n")
    f.write("Новые признаки (транзакции):\n")
    for f_name in new_features:
        f.write(f"  - {f_name}\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/features_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Feature Engineering завершён.")
print("=" * 60)
print(f"Датасет транзакций: {FEATURES_PATH}")
print(f"Датасет по месяцам: {MONTHLY_PATH}")
print(f"Отчёт:              {REPORTS_DIR}/features_report.txt")