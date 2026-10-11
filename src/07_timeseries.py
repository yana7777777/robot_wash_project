"""
07_timeseries.py — Временные ряды: прогноз прибыли и выручки.

Три ряда:
  1. Сырая прибыль (profit)
  2. Сглаженная прибыль (rolling mean 3 мес.)
  3. Выручка (income)

Методы: ARIMA, SARIMA, Prophet, LSTM.
Читает:  data/robot_wash_monthly.csv
Пишет:   reports/timeseries_report.txt
         reports/forecast_income_2027.csv
         models/prophet_model.pkl
         images/28-34*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 6)
sns.set_style("whitegrid")

MONTHLY_PATH = 'data/robot_wash_monthly.csv'
MODELS_DIR = 'models'
IMAGES_DIR = 'images'
REPORTS_DIR = 'reports'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

TEST_SIZE = 12


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_test_split_ts(series, test_size=TEST_SIZE):
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    return train, test


def run_arima(train, test, order=(2, 1, 2)):
    try:
        model = ARIMA(train, order=order)
        fit = model.fit()
        pred = fit.forecast(steps=len(test))
        pred.index = test.index
        return pred
    except Exception as e:
        print(f"  ARIMA ошибка: {e}")
        return None


def run_sarima(train, test, order=(1, 1, 1), seasonal=(1, 1, 1, 12)):
    try:
        model = SARIMAX(train, order=order, seasonal_order=seasonal)
        fit = model.fit(disp=False)
        pred = fit.forecast(steps=len(test))
        pred.index = test.index
        return pred
    except Exception as e:
        print(f"  SARIMA ошибка: {e}")
        return None


def run_prophet(train, test):
    """ИСПРАВЛЕНО: корректная подготовка DataFrame для Prophet."""
    try:
        prophet_train = train.reset_index()
        prophet_train.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False)
        model.fit(prophet_train)
        future = model.make_future_dataframe(periods=len(test), freq='MS')
        forecast = model.predict(future)
        pred = forecast.iloc[-len(test):]['yhat'].values
        pred_series = pd.Series(pred, index=test.index)
        return pred_series, model, forecast
    except Exception as e:
        print(f"  Prophet ошибка: {e}")
        return None, None, None


def run_lstm(train, test, lookback=12):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        full = pd.concat([train, test])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(full.values.reshape(-1, 1))

        X, y = [], []
        for i in range(lookback, len(scaled)):
            X.append(scaled[i - lookback:i, 0])
            y.append(scaled[i, 0])
        X, y = np.array(X), np.array(y)

        split = len(X) - len(test)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True,
                 input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=8,
                  verbose=0, callbacks=[es])

        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled).flatten()
        pred_series = pd.Series(pred, index=test.index)
        return pred_series, model
    except Exception as e:
        print(f"  LSTM ошибка: {e}")
        return None, None


def evaluate(name, test, pred):
    if pred is None:
        return {'Модель': name, 'MAE': None, 'RMSE': None, 'MAPE': None}
    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))
    mp = mape(test, pred)
    return {'Модель': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mp}


def analyze_series(name, series, images_prefix):
    """Полный цикл анализа одного ряда."""
    print("\n" + "=" * 70)
    print(f"АНАЛИЗ РЯДА: {name}")
    print("=" * 70)

    print(f"Месяцев: {len(series)}")
    print(f"Среднее: {series.mean():,.0f}")
    print(f"Мин: {series.min():,.0f} | Макс: {series.max():,.0f}")

    train, test = train_test_split_ts(series)
    print(f"Обучающая: {len(train)}, тестовая: {len(test)}")

    results = []

    # ARIMA
    print("\n  Обучение ARIMA...")
    arima_pred = run_arima(train, test)
    results.append(evaluate('ARIMA', test, arima_pred))

    # SARIMA
    print("  Обучение SARIMA...")
    sarima_pred = run_sarima(train, test)
    results.append(evaluate('SARIMA', test, sarima_pred))

    # Prophet
    print("  Обучение Prophet...")
    prophet_pred, prophet_model, prophet_forecast = run_prophet(train, test)
    results.append(evaluate('Prophet', test, prophet_pred))

    # LSTM
    print("  Обучение LSTM...")
    lstm_pred, lstm_model = run_lstm(train, test)
    results.append(evaluate('LSTM', test, lstm_pred))

    results_df = pd.DataFrame(results).dropna(subset=['MAPE'])
    print(f"\n  Результаты ({name}):")
    print(results_df.to_string(index=False))

    # График
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, test.values, marker='o', label='Реальные',
             color='black', linewidth=2.5, markersize=8)

    if arima_pred is not None:
        plt.plot(test.index, arima_pred, marker='s', label='ARIMA', alpha=0.7)
    if sarima_pred is not None:
        plt.plot(test.index, sarima_pred, marker='^', label='SARIMA', alpha=0.7)
    if prophet_pred is not None:
        plt.plot(test.index, prophet_pred, marker='D', label='Prophet', alpha=0.7)
    if lstm_pred is not None:
        plt.plot(test.index, lstm_pred, marker='v', label='LSTM', alpha=0.7)

    plt.title(f'Прогноз: {name}')
    plt.xlabel('Дата')
    plt.ylabel('Значение, руб.')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/{images_prefix}_models.png', dpi=150)
    plt.close()
    print(f"  Сохранён: {images_prefix}_models.png")

    return results_df, prophet_model, prophet_forecast


# ============================================================
# ШАГ 1. ЗАГРУЗКА
# ============================================================
print("=" * 70)
print("ШАГ 1. Загрузка помесячных данных")
print("=" * 70)

df = pd.read_csv(MONTHLY_PATH)
df['ds'] = pd.to_datetime(df['year_month'] + '-01')
df = df.sort_values('ds').reset_index(drop=True)

# 1. Сырая прибыль
profit_raw = df.set_index('ds')['profit'].copy()

# 2. Сглаженная прибыль (rolling mean 3 месяца)
profit_smooth = profit_raw.rolling(3).mean().dropna()

# 3. Выручка
income = df.set_index('ds')['income'].copy()

print(f"Всего месяцев: {len(df)}")
print(f"Период: {df['ds'].min().date()} — {df['ds'].max().date()}")


# ============================================================
# ШАГ 2. ВИЗУАЛИЗАЦИЯ ТРЁХ РЯДОВ
# ============================================================
print("\n" + "=" * 70)
print("ШАГ 2. Визуализация трёх рядов")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

axes[0].plot(profit_raw.index, profit_raw.values, color='steelblue')
axes[0].set_title('1. Сырая прибыль (высокая волатильность)')
axes[0].set_ylabel('Прибыль, руб.')
axes[0].grid(True, alpha=0.3)

axes[1].plot(profit_smooth.index, profit_smooth.values, color='orange')
axes[1].set_title('2. Сглаженная прибыль (rolling mean 3 мес.)')
axes[1].set_ylabel('Прибыль, руб.')
axes[1].grid(True, alpha=0.3)

axes[2].plot(income.index, income.values, color='green')
axes[2].set_title('3. Выручка (стабильный растущий ряд)')
axes[2].set_ylabel('Выручка, руб.')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/28_timeseries_overview.png', dpi=150)
plt.close()
print("Сохранён: 28_timeseries_overview.png")


# ============================================================
# ШАГ 3. АНАЛИЗ ТРЁХ РЯДОВ
# ============================================================
res_profit, pm_profit, pf_profit = analyze_series(
    'Сырая прибыль', profit_raw, '29_raw_profit')

res_smooth, pm_smooth, pf_smooth = analyze_series(
    'Сглаженная прибыль', profit_smooth, '30_smooth_profit')

res_income, pm_income, pf_income = analyze_series(
    'Выручка', income, '31_income')


# ============================================================
# ШАГ 4. СРАВНЕНИЕ ТРЁХ РЯДОВ
# ============================================================
print("\n" + "=" * 70)
print("ШАГ 4. Сравнение трёх рядов")
print("=" * 70)

# Сводная таблица по MAPE
summary = pd.DataFrame({
    'Сырая прибыль': res_profit.set_index('Модель')['MAPE'],
    'Сглаженная прибыль': res_smooth.set_index('Модель')['MAPE'],
    'Выручка': res_income.set_index('Модель')['MAPE'],
})
print("\nСводная таблица MAPE (%):")
print(summary.to_string())

# График сравнения
fig, ax = plt.subplots(figsize=(12, 6))
summary.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Сравнение MAPE для трёх рядов (чем меньше, тем лучше)')
ax.set_ylabel('MAPE, %')
ax.set_xlabel('Модель')
plt.xticks(rotation=0)
plt.legend(title='Ряд')
plt.tight_layout()
plt.savefig(f'{IMAGES_DIR}/32_timeseries_comparison.png', dpi=150)
plt.close()
print("Сохранён: 32_timeseries_comparison.png")


# ============================================================
# ШАГ 5. ПРОГНОЗ ВЫРУЧКИ НА БУДУЩИЙ ПЕРИОД (Prophet)
# ============================================================
print("\n" + "=" * 70)
print("ШАГ 5. Прогноз выручки на будущий период (Prophet)")
print("=" * 70)

if pm_income is not None:
    # Переобучаем Prophet на полной выручке
    prophet_full = income.reset_index()
    prophet_full.columns = ['ds', 'y']
    model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                         daily_seasonality=False)
    model_full.fit(prophet_full)

    future = model_full.make_future_dataframe(periods=24, freq='MS')
    forecast_future = model_full.predict(future)
    future_pred = forecast_future.iloc[-24:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    plt.figure(figsize=(14, 6))
    plt.plot(income.index, income.values, label='Историческая выручка', color='steelblue')
    plt.plot(future_pred['ds'], future_pred['yhat'], label='Прогноз Prophet',
             color='red', linewidth=2)
    plt.fill_between(future_pred['ds'], future_pred['yhat_lower'],
                     future_pred['yhat_upper'], alpha=0.2, color='red',
                     label='95% доверительный интервал')
    plt.title('Прогноз выручки на 24 месяца вперёд (Prophet)')
    plt.xlabel('Дата')
    plt.ylabel('Выручка, руб.')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMAGES_DIR}/33_income_forecast.png', dpi=150)
    plt.close()
    print("Сохранён: 33_income_forecast.png")

    future_pred.to_csv(f'{REPORTS_DIR}/forecast_income_2027.csv', index=False)
    print(f"Прогноз сохранён: {REPORTS_DIR}/forecast_income_2027.csv")

    avg_future = future_pred['yhat'].mean()
    print(f"\nСредняя прогнозная выручка на 24 мес. вперёд: {avg_future:,.0f} руб.")

# Сохраняем модель Prophet (для выручки)
if pm_income is not None:
    with open(f'{MODELS_DIR}/prophet_model.pkl', 'wb') as f:
        pickle.dump(pm_income, f)
    print(f"Модель Prophet сохранена: {MODELS_DIR}/prophet_model.pkl")


# ============================================================
# ШАГ 6. КОМПОНЕНТЫ PROPHET (выручка)
# ============================================================
if pf_income is not None:
    try:
        fig = pm_income.plot_components(pf_income)
        plt.tight_layout()
        plt.savefig(f'{IMAGES_DIR}/34_prophet_components.png', dpi=150)
        plt.close()
        print("Сохранён: 34_prophet_components.png")
    except Exception as e:
        print(f"Компоненты Prophet: {e}")


# ============================================================
# ШАГ 7. ОТЧЁТ
# ============================================================
print("\n" + "=" * 70)
print("Сохранение отчёта")
print("=" * 70)

with open(f'{REPORTS_DIR}/timeseries_report.txt', 'w', encoding='utf-8') as f:
    f.write("ОТЧЁТ: ВРЕМЕННЫЕ РЯДЫ (3 ряда)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Всего месяцев: {len(df)}\n")
    f.write(f"Период: {df['ds'].min().date()} — {df['ds'].max().date()}\n")
    f.write(f"Тестовая выборка: {TEST_SIZE} мес.\n\n")

    f.write("=" * 70 + "\n")
    f.write("1. СЫРАЯ ПРИБЫЛЬ\n")
    f.write("=" * 70 + "\n")
    f.write(f"Среднее: {profit_raw.mean():,.0f}\n")
    f.write(f"Мин: {profit_raw.min():,.0f} | Макс: {profit_raw.max():,.0f}\n\n")
    f.write(res_profit.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 70 + "\n")
    f.write("2. СГЛАЖЕННАЯ ПРИБЫЛЬ (rolling 3)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Среднее: {profit_smooth.mean():,.0f}\n")
    f.write(f"Мин: {profit_smooth.min():,.0f} | Макс: {profit_smooth.max():,.0f}\n\n")
    f.write(res_smooth.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 70 + "\n")
    f.write("3. ВЫРУЧКА\n")
    f.write("=" * 70 + "\n")
    f.write(f"Среднее: {income.mean():,.0f}\n")
    f.write(f"Мин: {income.min():,.0f} | Макс: {income.max():,.0f}\n\n")
    f.write(res_income.to_string(index=False))
    f.write("\n\n")

    f.write("=" * 70 + "\n")
    f.write("СВОДНАЯ ТАБЛИЦА MAPE (%)\n")
    f.write("=" * 70 + "\n")
    f.write(summary.to_string())

print(f"Отчёт сохранён: {REPORTS_DIR}/timeseries_report.txt")


# ============================================================
# ИТОГ
# ============================================================
print("\n" + "=" * 70)
print("ГОТОВО! Временные ряды завершены.")
print("=" * 70)
print("\nЛучшие модели по MAPE для каждого ряда:")
for name, res in [('Сырая прибыль', res_profit),
                  ('Сглаженная прибыль', res_smooth),
                  ('Выручка', res_income)]:
    if len(res) > 0:
        best = res.loc[res['MAPE'].idxmin()]
        print(f"  {name:25s}: {best['Модель']:10s} (MAPE = {best['MAPE']:.2f}%)")