"""
08_optimization.py — Оптимизация финансовых показателей.

Задачи:
  1. Линейная оптимизация закупок/продаж по месяцам (максимум прибыли).
  2. Оптимизация ценовой политики (максимум выручки при ограничении наценки).
  3. Сравнение методов: linprog, minimize (SLSQP), differential_evolution.

Читает:  data/robot_wash_monthly.csv
Пишет:   reports/optimization_report.txt
         images/35-37*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

from scipy.optimize import linprog, minimize, differential_evolution

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

MONTHLY_PATH = 'data/robot_wash_monthly.csv'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# ШАГ 1. ЗАГРУЗКА
# ============================================================
print("=" * 60)
print("ШАГ 1. Загрузка помесячных данных")
print("=" * 60)

df = pd.read_csv(MONTHLY_PATH)
print(f"Строк: {len(df)}, колонок: {len(df.columns)}")


# ============================================================
# ЗАДАЧА 1. ЛИНЕЙНАЯ ОПТИМИЗАЦИЯ: ПЛАН ПРОДАЖ
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 1. Линейная оптимизация: план продаж/закупок")
print("=" * 70)

# Берём годовые средние для упрощения — оптимизируем квартальный план
# Переменные: x1 = роботов продать, x2 = роботов закупить
# Максимизируем: revenue * x1 - cost * x2
# Ограничения:
#   x2 <= 200 (склад)
#   x1 <= x2 * 1.1 (нельзя продать больше, чем куплено)
#   x1 >= 50 (минимум продаж)
#   x1, x2 >= 0

avg_price = df['income'].sum() / max(df['quantity_sold'].sum(), 1)  # средняя цена робота
avg_cost = df['expense'].sum() / max(df['quantity_sold'].sum(), 1) * 0.55  # ~55% от цены

print(f"Средняя цена робота:    {avg_price:,.0f} руб.")
print(f"Средняя себестоимость:  {avg_cost:,.0f} руб.")

# linprog минимизирует → берём -прибыль
c = [-avg_price, avg_cost]

# Ограничения A_ub * x <= b_ub
A_ub = [
    [0, 1],       # x2 <= 200
    [1, -1.1],    # x1 - 1.1*x2 <= 0
    [-1, 0],      # -x1 <= -50
]
b_ub = [200, 0, -50]

bounds = [(0, None), (0, None)]

res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

if res_lp.success:
    x1_opt, x2_opt = res_lp.x
    profit_lp = -res_lp.fun
    print(f"\nОптимум (linprog):")
    print(f"  Продать роботов: {x1_opt:.1f}")
    print(f"  Закупить роботов: {x2_opt:.1f}")
    print(f"  Максимальная прибыль: {profit_lp:,.0f} руб.")
else:
    x1_opt, x2_opt, profit_lp = 50, 100, 0
    print("linprog не нашёл решения")


# ============================================================
# ЗАДАЧА 2. НЕЛИНЕЙНАЯ ОПТИМИЗАЦИЯ (SLSQP): ЦЕНОВАЯ ПОЛИТИКА
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 2. Нелинейная оптимизация: ценовая политика")
print("=" * 70)

# Функция спроса: qty = A * price^(-B), B=1.5 (эластичность)
base_qty = df['quantity_sold'].sum() / 12  # среднее за месяц
base_price = avg_price
elasticity = 1.5

def revenue(price):
    qty = base_qty * (price / base_price) ** (-elasticity)
    return -qty * price  # минус для минимизации

# Оптимизация цены
res_nl = minimize(revenue, x0=[base_price],
                  bounds=[(base_price * 0.5, base_price * 2.0)],
                  method='SLSQP')

if res_nl.success:
    optimal_price = res_nl.x[0]
    optimal_qty = base_qty * (optimal_price / base_price) ** (-elasticity)
    optimal_revenue = optimal_price * optimal_qty
    print(f"Оптимальная цена (SLSQP): {optimal_price:,.0f} руб.")
    print(f"Ожидаемый объём продаж:   {optimal_qty:.1f} шт./мес")
    print(f"Ожидаемая выручка:        {optimal_revenue:,.0f} руб./мес")
else:
    optimal_price = base_price
    optimal_qty = base_qty
    optimal_revenue = base_price * base_qty


# ============================================================
# ЗАДАЧА 3. ГЛОБАЛЬНАЯ ОПТИМИЗАЦИЯ (differential_evolution)
# ============================================================
print("\n" + "=" * 70)
print("ЗАДАЧА 3. Глобальная оптимизация: 3 параметра")
print("=" * 70)

# Параметры: доля наценки, доля логистики, объём закупки
# Целевая: максимум прибыли от квартала
def objective(params):
    markup, logistics_share, purchase_vol = params
    # Выручка = объём продаж * цена * наценка
    sales = purchase_vol * 0.9  # 90% распродаётся
    rev = sales * avg_price * markup
    # Расходы = закупка + логистика + налоги
    cost = purchase_vol * avg_cost + rev * logistics_share
    return -(rev - cost)  # минус — для минимизации

bounds = [(1.0, 1.5), (0.05, 0.30), (100, 500)]

res_de = differential_evolution(objective, bounds, seed=42, maxiter=200)

if res_de.success:
    best_markup, best_log, best_vol = res_de.x
    best_profit = -res_de.fun
    print(f"Оптимум (differential_evolution):")
    print(f"  Наценка:          {best_markup:.3f}")
    print(f"  Доля логистики:   {best_log:.3f}")
    print(f"  Объём закупки:    {best_vol:.1f} шт.")
    print(f"  Максимум прибыли: {best_profit:,.0f} руб.")


# ============================================================
# ГРАФИКИ
# ============================================================
print("\n" + "=" * 60)
print("Построение графиков")
print("=" * 60)

# 35. Сравнение методов оптимизации
plt.figure(figsize=(10, 6))
methods = ['linprog\n(задачи 1)', 'SLSQP\n(задача 2)', 'diff.evolution\n(задача 3)']
profits = [profit_lp / 1e6, optimal_revenue / 1e6, best_profit / 1e6]
colors = ['steelblue', 'coral', 'teal']
bars = plt.bar(methods, profits, color=colors)
for bar, val in zip(bars, profits):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f'{val:.1f}M', ha='center', fontsize=12, fontweight='bold')
plt.title('Сравнение результатов методов оптимизации')
plt.ylabel('Целевая функция, млн руб.')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/35_optimization_methods.png', dpi=150)
plt.close()
print("Сохранён: 35_optimization_methods.png")

# 36. Кривая спроса vs оптимальная цена
prices = np.linspace(base_price * 0.5, base_price * 2.0, 100)
revenues = [base_qty * (p / base_price) ** (-elasticity) * p / 1e6 for p in prices]

plt.figure(figsize=(12, 6))
plt.plot(prices / 1e6, revenues, color='steelblue', linewidth=2)
plt.axvline(optimal_price / 1e6, color='red', linestyle='--',
            label=f'Оптимум: {optimal_price/1e6:.2f} млн руб.')
plt.title('Кривая выручки от цены (эластичность 1.5)')
plt.xlabel('Цена робота, млн руб.')
plt.ylabel('Выручка, млн руб./мес')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/36_optimization_price.png', dpi=150)
plt.close()
print("Сохранён: 36_optimization_price.png")

# 37. Факт vs оптимум по месяцам
monthly_opt = df[['year_month', 'profit']].copy()
monthly_opt['profit_opt'] = monthly_opt['profit'] * (profit_lp / max(df['profit'].sum(), 1))

plt.figure(figsize=(14, 6))
plt.plot(monthly_opt['year_month'], monthly_opt['profit'] / 1e6,
         label='Факт', color='steelblue', linewidth=2)
plt.plot(monthly_opt['year_month'], monthly_opt['profit_opt'] / 1e6,
         label='Оптимум (линейная модель)', color='red', linestyle='--', linewidth=2)
plt.title('Прибыль: факт vs оптимизированный план')
plt.xlabel('Месяц')
plt.ylabel('Прибыль, млн руб.')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/37_optimization_compare.png', dpi=150)
plt.close()
print("Сохранён: 37_optimization_compare.png")


# ============================================================
# ОТЧЁТ
# ============================================================
print("\n" + "=" * 60)
print("Сохранение отчёта")
print("=" * 60)

with open(f'{REPORTS_DIR}/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: ОПТИМИЗАЦИЯ ФИНАНСОВЫХ ПОКАЗАТЕЛЕЙ\n")
    f.write("=" * 70 + "\n\n")

    f.write("ЗАДАЧА 1. Линейная оптимизация (linprog)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Переменные: продать (x1), закупить (x2)\n")
    f.write(f"Оптимум: x1 = {x1_opt:.1f}, x2 = {x2_opt:.1f}\n")
    f.write(f"Максимальная прибыль: {profit_lp:,.0f} руб.\n\n")

    f.write("ЗАДАЧА 2. Нелинейная оптимизация цены (SLSQP)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Эластичность спроса: {elasticity}\n")
    f.write(f"Оптимальная цена: {optimal_price:,.0f} руб.\n")
    f.write(f"Ожидаемая выручка: {optimal_revenue:,.0f} руб./мес\n\n")

    f.write("ЗАДАЧА 3. Глобальная оптимизация (differential_evolution)\n")
    f.write("-" * 70 + "\n")
    f.write(f"Наценка: {best_markup:.3f}\n")
    f.write(f"Доля логистики: {best_log:.3f}\n")
    f.write(f"Объём закупки: {best_vol:.1f} шт.\n")
    f.write(f"Максимум прибыли: {best_profit:,.0f} руб.\n")

print(f"Отчёт сохранён: {REPORTS_DIR}/optimization_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 60)
print("ГОТОВО! Оптимизация завершена.")
print("=" * 60)
print(f"linprog:  прибыль = {profit_lp:,.0f} руб.")
print(f"SLSQP:    оптимум цены = {optimal_price:,.0f} руб.")
print(f"diff.evo: прибыль = {best_profit:,.0f} руб.")