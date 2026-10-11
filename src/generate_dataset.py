"""
Генератор датасета robot_wash_extended.csv (v7)

Ключевые принципы:
  - ~7000 чистых строк + 100–200 грязных
  - 2-4 партии закупок в месяц (по 8-12 роботов)
  - 10-16 сделок продаж в месяц (по 2-4 робота)
  - Цены в юанях, продажа = закупка × индивидуальная наценка (1.35–1.55)
  - Сезонность: пик продаж весной/осенью
  - Тренд: инфляция Китая 2% в год
  - НДС: 18% (2015–2018), 20% (2019–2025), 22% (с 2026)
  - Сохраняет сразу в data/robot_wash_extended.csv
"""

import pandas as pd
import random
from datetime import datetime
import os

random.seed(42)

START_YEAR = 2015
END_YEAR = 2026
END_MONTH = 10

legal_entities = ['ООО Ветерок', 'ООО Ромашка', 'ИП Иванов И.И.', 'ООО Заря']

counterparties_sale = [
    'ООО Мойка', 'ООО Луна', 'ООО Автомойка-Юг', 'ООО Мойка-Север',
    'ООО Чистый Город', 'ООО ТехРегионЭнерго', 'ООО СТД Групп',
    'ООО Заволжье', 'ИП Быков', 'ИП Петров', 'ООО Авто-Сервис',
    'ООО АкваПром', 'ООО Север-Мойка', 'ИП Сидоров', 'ООО ЭкоМойка',
    'ООО Сити-Сервис', 'ООО Гранд-Мойка', 'ИП Кузнецов',
    'ООО МегаМойка', 'ООО Урал-Сервис'
]

months_ru = {1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель', 5: 'Май', 6: 'Июнь',
             7: 'Июль', 8: 'Август', 9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'}

EXCHANGE_RATES = {
    2015: 6.20, 2016: 6.35, 2017: 6.50, 2018: 6.75,
    2019: 6.95, 2020: 7.10, 2021: 7.30, 2022: 7.50,
    2023: 8.18, 2024: 8.18, 2025: 9.00, 2026: 9.50
}

# Цены в ЮАНЯХ
CNY_COST_PLUS = 246_000
CNY_COST_360 = 197_000
CNY_PRICE_PLUS = 331_000
CNY_PRICE_360 = 268_000

CN_INFLATION = 0.02

CNY_MONTAGE = 21_000
CNY_SERVICE = 6_300
CNY_SPARE_UNIT = 1_130
CNY_DELIVERY = 57_600
CNY_CUSTOMS_FEE = 693

# Сезонные веса
SEASONAL_WEIGHTS = {
    1: 0.75, 2: 0.90, 3: 1.30, 4: 1.45, 5: 1.25,
    6: 1.00, 7: 0.75, 8: 0.85, 9: 1.40,
    10: 1.30, 11: 1.10, 12: 0.90
}


def _build_client_markups():
    rnd = random.Random(777)
    markups = {}
    for client in counterparties_sale:
        markups[client] = round(rnd.uniform(1.35, 1.55), 3)
    return markups

CLIENT_MARKUPS = _build_client_markups()


def noise(pct=0.02):
    return random.uniform(1 - pct, 1 + pct)


def get_vat_rate(year):
    if year >= 2026:
        return 22
    elif year >= 2019:
        return 20
    return 18


def get_cny_inflation(year):
    return (1 + CN_INFLATION) ** (year - 2020)


def get_cny_cost(year, model):
    base = CNY_COST_PLUS if model == 'Leisuwash 360 Plus' else CNY_COST_360
    return base * get_cny_inflation(year)


def get_cny_price_for_client(year, model, client):
    cost_cny = get_cny_cost(year, model)
    markup = CLIENT_MARKUPS.get(client, 1.45)
    return cost_cny * markup


def get_exchange_rate(year):
    return round(EXCHANGE_RATES[year] * noise(0.01), 2)


def get_season(month):
    return SEASONAL_WEIGHTS.get(month, 1.0)


def cny_to_rub(cny_amount, rate):
    return round(cny_amount * rate)


def base_row(date):
    return {
        'date': date.strftime('%Y-%m-%d'),
        'year': date.year,
        'month': date.month,
        'month_name': months_ru[date.month],
        'quarter': f'Q{(date.month - 1) // 3 + 1}',
        'legal_entity': random.choice(legal_entities),
        'counterparty': '',
        'product_model': '',
        'quantity': 0,
        'income_article': '',
        'income_group': '',
        'income': 0,
        'expense_article': '',
        'expense_group': '',
        'expense': 0,
        'exchange_rate': get_exchange_rate(date.year),
        'tnved_code': '',
        'customs_duty_rate': 0,
        'customs_duty_amount': 0,
        'vat_rate': 0,
        'vat_amount': 0,
        'customs_fee': 0,
        'bank_commission': 0,
        'delivery_cost_rub': 0,
        'notes': ''
    }


def make_income_row(date, article):
    row = base_row(date)
    year = date.year
    rate = row['exchange_rate']

    client = random.choice(counterparties_sale)
    row['counterparty'] = client
    row['income_article'] = article
    row['income_group'] = 'Выручка'

    if article == 'Продажа робота':
        model = random.choices(['Leisuwash 360 Plus', 'Leisuwash 360'],
                                weights=[0.6, 0.4])[0]
        qty = random.choice([2, 3, 3, 4])
        cny_unit = get_cny_price_for_client(year, model, client) * noise(0.02)
        row['product_model'] = model
        row['quantity'] = qty
        row['income'] = cny_to_rub(qty * cny_unit, rate)
        row['tnved_code'] = '8424890001'
        row['customs_duty_rate'] = 8
        row['customs_duty_amount'] = round(row['income'] * 0.08)
        row['vat_rate'] = get_vat_rate(year)
        row['vat_amount'] = round(row['income'] * row['vat_rate'] / 100)
        row['notes'] = 'Продажа'

    elif article == 'Монтаж':
        cny_amount = CNY_MONTAGE * get_cny_inflation(year) * noise(0.03)
        row['income'] = cny_to_rub(cny_amount, rate)
        row['notes'] = 'Монтаж'

    elif article == 'Продажа запчастей':
        row['product_model'] = 'Запчасти'
        qty = random.choice([3, 4, 5])
        cny_unit = CNY_SPARE_UNIT * get_cny_inflation(year) * noise(0.03)
        row['quantity'] = qty
        row['income'] = cny_to_rub(qty * cny_unit, rate)
        row['notes'] = 'Запчасти'

    elif article == 'Сервисное обслуживание':
        cny_amount = CNY_SERVICE * get_cny_inflation(year) * noise(0.03)
        row['income'] = cny_to_rub(cny_amount, rate)
        row['notes'] = 'Сервис'

    return row


def make_expense_row(date, article):
    row = base_row(date)
    year = date.year
    rate = row['exchange_rate']

    if article == 'Закупка товара':
        model = random.choices(['Leisuwash 360 Plus', 'Leisuwash 360'],
                                weights=[0.6, 0.4])[0]
        qty = random.choice([8, 10, 10, 12])  # крупные партии
        cny_unit = get_cny_cost(year, model) * noise(0.01)
        row['counterparty'] = 'Leisuwash China'
        row['product_model'] = model
        row['quantity'] = qty
        row['expense_article'] = article
        row['expense_group'] = 'Себестоимость'
        row['expense'] = cny_to_rub(qty * cny_unit, rate)
        row['tnved_code'] = '8424890001'
        row['customs_duty_rate'] = 8
        row['customs_duty_amount'] = round(row['expense'] * 0.08)
        row['vat_rate'] = get_vat_rate(year)
        row['vat_amount'] = round((row['expense'] + row['customs_duty_amount']) * row['vat_rate'] / 100)
        row['customs_fee'] = cny_to_rub(CNY_CUSTOMS_FEE * get_cny_inflation(year), rate)
        row['bank_commission'] = round(row['expense'] * 0.005)
        row['delivery_cost_rub'] = cny_to_rub(CNY_DELIVERY * get_cny_inflation(year), rate)
        row['notes'] = f'Закупка {model}'

    elif article == 'Доставка':
        row['counterparty'] = 'ТрансЛогистик'
        row['expense_article'] = article
        row['expense_group'] = 'Логистика'
        cny_amount = CNY_DELIVERY * get_cny_inflation(year) * noise(0.02)
        row['expense'] = cny_to_rub(cny_amount, rate)
        row['delivery_cost_rub'] = row['expense']
        row['notes'] = "Контейнер 40'"

    elif article == 'Таможенный сбор':
        row['counterparty'] = 'ФТС'
        row['expense_article'] = article
        row['expense_group'] = 'Таможня'
        row['expense'] = cny_to_rub(CNY_CUSTOMS_FEE * get_cny_inflation(year), rate)
        row['customs_fee'] = row['expense']
        row['notes'] = 'Сбор'

    elif article == 'Таможенная пошлина':
        row['counterparty'] = 'ФТС'
        row['expense_article'] = article
        row['expense_group'] = 'Таможня'
        avg_cny = 10 * get_cny_cost(year, 'Leisuwash 360 Plus')
        cny_amount = avg_cny * 0.08 * noise(0.02)
        row['expense'] = cny_to_rub(cny_amount, rate)
        row['tnved_code'] = '8424890001'
        row['customs_duty_rate'] = 8
        row['customs_duty_amount'] = row['expense']
        row['notes'] = 'Пошлина 8%'

    elif article == 'НДС при импорте':
        row['counterparty'] = 'ФТС'
        row['expense_article'] = article
        row['expense_group'] = 'Таможня'
        vat_rate = get_vat_rate(year)
        avg_cny = 10 * get_cny_cost(year, 'Leisuwash 360 Plus')
        cny_amount = avg_cny * 1.08 * vat_rate / 100 * noise(0.02)
        row['expense'] = cny_to_rub(cny_amount, rate)
        row['tnved_code'] = '8424890001'
        row['customs_duty_rate'] = 8
        row['vat_rate'] = vat_rate
        row['customs_duty_amount'] = round(row['expense'] / (vat_rate / 100) * 0.08)
        row['vat_amount'] = row['expense']
        row['notes'] = f'НДС {vat_rate}%'

    elif article == 'Комиссия банка':
        row['counterparty'] = 'Банк Т-Банк'
        row['expense_article'] = article
        row['expense_group'] = 'Банк'
        avg_cny = 10 * get_cny_cost(year, 'Leisuwash 360 Plus')
        cny_amount = avg_cny * 0.005 * noise(0.03)
        row['expense'] = cny_to_rub(cny_amount, rate)
        row['bank_commission'] = row['expense']
        row['notes'] = 'Комиссия 0.5%'

    return row


def generate_month(year, month):
    season = get_season(month)
    days_in_month = 28
    rows = []

    def rand_date():
        return datetime(year, month, random.randint(1, days_in_month))

    # --- ДОХОДЫ ---
    # 10-16 сделок продажи роботов в месяц (удвоено для ~7000 строк)
    n_sales = max(10, round(12 * season))
    for _ in range(n_sales):
        rows.append(make_income_row(rand_date(), 'Продажа робота'))
        rows.append(make_income_row(rand_date(), 'Монтаж'))

    n_service = max(4, round(4 * season))
    for _ in range(n_service):
        rows.append(make_income_row(rand_date(), 'Сервисное обслуживание'))
        rows.append(make_income_row(rand_date(), 'Продажа запчастей'))

    # --- РАСХОДЫ ---
    # 2-4 партии закупки в месяц (удвоено)
    n_purchases = max(2, round(1.6 * season))
    for _ in range(n_purchases):
        rows.append(make_expense_row(rand_date(), 'Закупка товара'))
        rows.append(make_expense_row(rand_date(), 'Доставка'))
        rows.append(make_expense_row(rand_date(), 'Таможенный сбор'))
        rows.append(make_expense_row(rand_date(), 'Таможенная пошлина'))
        rows.append(make_expense_row(rand_date(), 'НДС при импорте'))
        rows.append(make_expense_row(rand_date(), 'Комиссия банка'))

    return rows


def make_dirty_row(row):
    row = row.copy()
    issues = random.sample(range(1, 7), random.randint(1, 2))
    # Сортируем, чтобы issue 1 (None) шёл последним — иначе он ломает abs() и т.д.
    issues = sorted(issues, key=lambda x: x == 1)
    for issue in issues:
        if issue == 1:
            col = random.choice(list(row.keys()))
            if col not in ['date', 'year', 'month']:
                row[col] = None
        elif issue == 2:
            row['date'] = '2023-13-45'
        elif issue == 3:
            # защита от None
            if row['quantity'] is not None and row['quantity'] != 0:
                row['quantity'] = -abs(row['quantity'])
        elif issue == 4:
            row['income'] = 'N/A'
        elif issue == 5:
            row['exchange_rate'] = 'abc'
        elif issue == 6:
            row['tnved_code'] = '9999999999'
    return row


# ============================================================
# ОСНОВНОЙ ЦИКЛ
# ============================================================
print("Генерация датасета v7 (сбалансированный, ~7000 строк)...")
print(f"Период: {START_YEAR}-01 — {END_YEAR}-{END_MONTH:02d}")

all_rows = []
year = START_YEAR
month = 1
while (year < END_YEAR) or (year == END_YEAR and month <= END_MONTH):
    all_rows.extend(generate_month(year, month))
    month += 1
    if month > 12:
        month = 1
        year += 1

print(f"Сгенерировано чистых строк: {len(all_rows)}")

N_DIRTY = random.randint(100, 200)
print(f"Грязных строк: {N_DIRTY}")
dirty_rows = [make_dirty_row(random.choice(all_rows)) for _ in range(N_DIRTY)]
all_rows.extend(dirty_rows)
random.shuffle(all_rows)

cols = ['date', 'year', 'month', 'month_name', 'quarter', 'legal_entity', 'counterparty',
        'product_model', 'quantity', 'income_article', 'income_group', 'income',
        'expense_article', 'expense_group', 'expense', 'exchange_rate', 'tnved_code',
        'customs_duty_rate', 'customs_duty_amount', 'vat_rate', 'vat_amount',
        'customs_fee', 'bank_commission', 'delivery_cost_rub', 'notes']

df = pd.DataFrame(all_rows)[cols]
df = df.sort_values('date').reset_index(drop=True)

# Сохраняем сразу в data/
os.makedirs('data', exist_ok=True)
df.to_csv('data/robot_wash_extended.csv', index=False)

print(f"\nДатасет сохранён: data/robot_wash_extended.csv")
print(f"Всего строк: {len(df)} (чистых: {len(df) - N_DIRTY}, грязных: {N_DIRTY})")
print(f"Диапазон дат: {df['date'].min()} — {df['date'].max()}")

# ---- Проверка цен ----
print("\n" + "=" * 70)
print("ЦЕНЫ (в юанях):")
print("=" * 70)
print(f"{'Год':<6}{'Курс':<10}{'Закупка CNY':<15}{'Продажа CNY':<15}")
print("-" * 70)
for y in range(2015, 2027):
    rate = EXCHANGE_RATES[y]
    cny_c = get_cny_cost(y, 'Leisuwash 360 Plus')
    cny_p = cny_c * 1.45
    print(f"{y:<6}{rate:<10.2f}{cny_c:<15,.0f}{cny_p:<15,.0f}")

print("\n" + "=" * 70)
print("ОЖИДАЕМЫЙ БАЛАНС (месяц высокого сезона):")
print("=" * 70)
print("  Продаж:   ~12 сделок × 3 робота = 36 роботов")
print("  Закупок:  ~2 партии × 10 роботов = 20 роботов")
print("  → Выручка должна превышать расходы ✅")