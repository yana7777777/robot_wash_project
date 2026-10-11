import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

start_date = datetime(2022, 1, 1)
end_date = datetime(2026, 10, 31)

legal_entities = ['ООО Ветерок', 'ООО Ромашка', 'ИП Иванов И.И.', 'ООО Заря']
counterparties_purchase = ['Leisuwash China', 'ТрансЛогистик', 'ФТС', 'Банк Т-Банк']
counterparties_sale = ['ООО Мойка', 'ООО Луна', 'ООО Автомойка-Юг', 'ООО Мойка-Север',
                       'ООО Чистый Город', 'ООО ТехРегионЭнерго', 'ООО СТД Групп',
                       'ООО Заволжье', 'ИП Быков', 'ИП Петров', 'ООО Авто-Сервис']
product_models = ['Leisuwash 360 Plus', 'Leisuwash 360', 'Запчасти', '']
income_articles = ['Продажа робота', 'Монтаж', 'Продажа запчастей', 'Сервисное обслуживание']
expense_articles = ['Закупка товара', 'Доставка', 'Таможенный сбор',
                    'Таможенная пошлина', 'НДС при импорте', 'Комиссия банка']

months_ru = {1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель', 5: 'Май', 6: 'Июнь',
             7: 'Июль', 8: 'Август', 9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'}

exchange_rates = {2022: 7.5, 2023: 8.18, 2024: 8.18, 2025: 9.0, 2026: 9.5}


def get_exchange_rate(date):
    year = date.year
    base = exchange_rates.get(year, 8.18)
    return round(base + random.uniform(-0.2, 0.2), 2)


def get_vat_rate(date):
    """НДС: 20% до 2026 года, 22% с 1 января 2026."""
    if date.year >= 2026:
        return 22
    return 20


def generate_clean_row():
    days_diff = (end_date - start_date).days
    random_days = random.randint(0, days_diff)
    date = start_date + timedelta(days=random_days)
    year = date.year
    month = date.month
    month_name = months_ru[month]
    quarter = f'Q{(month - 1) // 3 + 1}'
    legal_entity = random.choice(legal_entities)
    is_expense = random.random() < 0.5

    if is_expense:
        expense_article = random.choice(expense_articles)
        if expense_article == 'Закупка товара':
            counterparty = 'Leisuwash China'
            product_model = random.choice(['Leisuwash 360 Plus', 'Leisuwash 360', 'Запчасти'])
            if product_model == 'Запчасти':
                quantity = random.randint(1, 20)
                expense = random.randint(50000, 300000)
                tnved_code = ''
                customs_duty_rate = 0
            else:
                quantity = random.randint(1, 10)
                unit_price = random.randint(1700000, 2000000) if product_model == 'Leisuwash 360 Plus' else random.randint(1300000, 1600000)
                expense = quantity * unit_price
                tnved_code = '8424890001'
                customs_duty_rate = 8
            customs_duty_amount = round(expense * customs_duty_rate / 100) if customs_duty_rate else 0
            vat_rate = get_vat_rate(date) if product_model != 'Запчасти' else 0
            vat_amount = round((expense + customs_duty_amount) * vat_rate / 100) if vat_rate else 0
            customs_fee = 4924 if product_model != 'Запчасти' else 0
            bank_commission = round(expense * 0.005)
            delivery_cost_rub = 409000 if product_model != 'Запчасти' else 0
            notes = f'Закупка {product_model}'
        elif expense_article == 'Доставка':
            counterparty = 'ТрансЛогистик'
            product_model = ''
            quantity = 0
            expense = 409000
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 409000
            notes = "Контейнер 40'"
        elif expense_article == 'Таможенный сбор':
            counterparty = 'ФТС'
            product_model = ''
            quantity = 0
            expense = 4924
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 4924
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Сбор'
        elif expense_article == 'Таможенная пошлина':
            counterparty = 'ФТС'
            product_model = ''
            quantity = 0
            expense = random.randint(100000, 1000000)
            tnved_code = '8424890001'
            customs_duty_rate = 8
            customs_duty_amount = expense
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Пошлина 8%'
        elif expense_article == 'НДС при импорте':
            counterparty = 'ФТС'
            product_model = ''
            quantity = 0
            expense = random.randint(200000, 2500000)
            tnved_code = '8424890001'
            customs_duty_rate = 8
            vat_rate = get_vat_rate(date)
            customs_duty_amount = round(expense / (vat_rate / 100) * 0.08)
            vat_amount = expense
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = f'НДС {vat_rate}%'
        elif expense_article == 'Комиссия банка':
            counterparty = 'Банк Т-Банк'
            product_model = ''
            quantity = 0
            expense = random.randint(5000, 60000)
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = expense
            delivery_cost_rub = 0
            notes = 'Комиссия 0.5%'
        else:
            counterparty = ''
            product_model = ''
            quantity = 0
            expense = 0
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = ''

        income_article = ''
        income_group = ''
        income = 0
        expense_group = ''
        if expense_article == 'Закупка товара':
            expense_group = 'Себестоимость'
        elif expense_article == 'Доставка':
            expense_group = 'Логистика'
        elif expense_article in ['Таможенный сбор', 'Таможенная пошлина', 'НДС при импорте']:
            expense_group = 'Таможня'
        elif expense_article == 'Комиссия банка':
            expense_group = 'Банк'

    else:
        income_article = random.choice(income_articles)
        if income_article == 'Продажа робота':
            counterparty = random.choice(counterparties_sale)
            product_model = random.choice(['Leisuwash 360 Plus', 'Leisuwash 360'])
            quantity = random.randint(1, 5)
            unit_price = random.randint(2200000, 2500000) if product_model == 'Leisuwash 360 Plus' else random.randint(1800000, 2000000)
            income = quantity * unit_price
            tnved_code = '8424890001'
            customs_duty_rate = 8
            customs_duty_amount = round(income * 0.08)
            vat_rate = get_vat_rate(date)
            vat_amount = round(income * vat_rate / 100)
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Продажа'
        elif income_article == 'Монтаж':
            counterparty = random.choice(counterparties_sale)
            product_model = ''
            quantity = 0
            income = 150000
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Монтаж'
        elif income_article == 'Продажа запчастей':
            counterparty = random.choice(counterparties_sale)
            product_model = 'Запчасти'
            quantity = random.randint(1, 10)
            income = quantity * random.randint(5000, 12000)
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Запчасти'
        elif income_article == 'Сервисное обслуживание':
            counterparty = random.choice(counterparties_sale)
            product_model = ''
            quantity = 0
            income = random.randint(30000, 60000)
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = 'Сервис'
        else:
            counterparty = ''
            product_model = ''
            quantity = 0
            income = 0
            tnved_code = ''
            customs_duty_rate = 0
            customs_duty_amount = 0
            vat_rate = 0
            vat_amount = 0
            customs_fee = 0
            bank_commission = 0
            delivery_cost_rub = 0
            notes = ''

        expense_article = ''
        expense_group = ''
        expense = 0
        income_group = 'Выручка'

    exchange_rate = get_exchange_rate(date)

    return {
        'date': date.strftime('%Y-%m-%d'),
        'year': year,
        'month': month,
        'month_name': month_name,
        'quarter': quarter,
        'legal_entity': legal_entity,
        'counterparty': counterparty,
        'product_model': product_model,
        'quantity': quantity,
        'income_article': income_article,
        'income_group': income_group,
        'income': income,
        'expense_article': expense_article,
        'expense_group': expense_group,
        'expense': expense,
        'exchange_rate': exchange_rate,
        'tnved_code': tnved_code,
        'customs_duty_rate': customs_duty_rate,
        'customs_duty_amount': customs_duty_amount,
        'vat_rate': vat_rate,
        'vat_amount': vat_amount,
        'customs_fee': customs_fee,
        'bank_commission': bank_commission,
        'delivery_cost_rub': delivery_cost_rub,
        'notes': notes
    }


def generate_dirty_row():
    row = generate_clean_row()
    issues = random.sample(range(1, 8), random.randint(1, 3))
    for issue in issues:
        if issue == 1:
            col = random.choice(list(row.keys()))
            if col not in ['date', 'year', 'month']:
                row[col] = None
        elif issue == 2:
            row['date'] = '01/01/2023'
        elif issue == 3:
            if row['quantity'] != 0:
                row['quantity'] = -abs(row['quantity'])
        elif issue == 4:
            row['income'] = 'N/A'
        elif issue == 5:
            row['exchange_rate'] = 'abc'
        elif issue == 6:
            row['tnved_code'] = '9999999999'
        elif issue == 7:
            pass
    return row


clean_rows = [generate_clean_row() for _ in range(300)]
dirty_rows = [generate_dirty_row() for _ in range(50)]
all_rows = clean_rows + dirty_rows
random.shuffle(all_rows)

df = pd.DataFrame(all_rows)
cols = ['date', 'year', 'month', 'month_name', 'quarter', 'legal_entity', 'counterparty',
        'product_model', 'quantity', 'income_article', 'income_group', 'income',
        'expense_article', 'expense_group', 'expense', 'exchange_rate', 'tnved_code',
        'customs_duty_rate', 'customs_duty_amount', 'vat_rate', 'vat_amount',
        'customs_fee', 'bank_commission', 'delivery_cost_rub', 'notes']
df = df[cols]

df.to_csv('robot_wash_extended.csv', index=False)
print("Датасет создан: robot_wash_extended.csv")
print(f"Всего строк: {len(df)} (чистых: 300, грязных: 50)")