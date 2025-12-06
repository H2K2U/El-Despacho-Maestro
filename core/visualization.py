# File: core/visualization.py
"""
Визуализация результатов оптимизации (Matplotlib + PySide6).

Зачем этот модуль нужен:
1) EnergyChart — "основной" график покрытия нагрузки за 24 часа:
   — вверх (положительное): генерация МГЭС/ВЭУ/ДЭС + разряд АКБ на нагрузку;
   — вниз (отрицательное): заряд АКБ (от ВИЭ или от ДЭС) + сброс/балласт (dump);
   — поверх всего: линия нагрузки.
2) BatteryChart — "диагностика" АКБ за 24 часа:
   — SoC (энергия, кВт*ч) по часам + линии SOC min/max;
   — мощности заряда/разряда (кВт) отдельным графиком.

Ключевая идея про единообразие:
— Вся логика выделения рядов "заряд/разряд" вынесена в core.battery_metrics.compute_charge_discharge().
  Это сделано специально, чтобы UI НЕ дублировал "как считать заряд" и не рассинхронизировался с тестами.
— Баланс энергии АКБ (SOC-side) считается через core.battery_metrics.compute_battery_energy_balance().

Важно про соглашения по данным schedule (OptimizationResult.schedule):
Ожидаемые колонки (не все обязательны):
— load: нагрузка, кВт;
— hydro: МГЭС, кВт;
— wind: ВЭУ, кВт;
— diesel: суммарная ДЭС, кВт (или вместо этого — колонки по агрегатам);
— dump: сброс/балласт, кВт;
— unserved: непокрытая нагрузка, кВт (если есть);
— soc / soc_kwh: SoC в кВт*ч;
— charge_power / discharge_power: агрегированные заряд/разряд АКБ (кВт), если алгоритм их пишет;
— battery_charge_from_renewable / battery_charge_from_diesel (и синонимы *_RES, *_DIESEL) — детальные компоненты заряда.

Про "колонки ДЭС по агрегатам":
— В greedy у тебя schedule_row дополнительно содержит пары "имя агрегата" -> мощность;
— Здесь мы отличаем агрегатные колонки от "системных" через known_columns.
  Все неизвестное считаем агрегатами ДЭС.
"""

import matplotlib.pyplot as plt
from PySide6.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import math

from core.battery_metrics import compute_battery_energy_balance, compute_charge_discharge


class EnergyChart(FigureCanvas):
    """
    График покрытия нагрузки "стаканом" (stacked bars) + линия нагрузки.

    Координатная система:
    — y > 0: генерация и разряд АКБ, которые покрывают нагрузку;
    — y < 0: заряд АКБ и сброс (dump), то есть "потребление/поглощение" мощности;
    — линия y=0 рисуется явно, чтобы визуально отделить "покрытие" от "поглощения".

    Почему так:
    — это самый читаемый способ показать баланс: что именно покрывает нагрузку,
      и куда девается избыток (заряд/сброс).
    """

    def __init__(self, parent=None, width=10, height=6, dpi=100, show_debug_series: bool = False):
        # Figure — контейнер Matplotlib. FigureCanvas — Qt-виджет, который умеет это рисовать.
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        # Политика размеров:
        # Expanding/Expanding — виджет будет растягиваться при ресайзе окна.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Минимальный размер, чтобы легенда/оси не превращались в кашу.
        self.setMinimumSize(400, 300)

        # Одна ось (subplot) на весь график покрытия нагрузки.
        self.ax = self.fig.add_subplot(111)

        # Флажок "показывать служебные ряды".
        # Сейчас в коде не используется, но оставлен для дальнейшей отладки:
        # например показывать residual/unserved/diesel_excess отдельным штрихом.
        self.show_debug_series = show_debug_series

        # Отступ справа под легенду:
        # right=0.75 означает, что поле графика занимает 75% ширины,
        # а остальное место отдаем под легенду (bbox_to_anchor справа).
        self.fig.subplots_adjust(left=0.1, right=0.75, top=0.95, bottom=0.1)

    def auto_scale_axes(self, load, bottom, charge_arrays, dump):
        """
        Автоматическое масштабирование оси Y.

        Входы:
        — load: массив нагрузки (кВт) за сутки;
        — bottom: "верхняя граница" stacked-bar после всех положительных столбцов;
                  фактически это суммарная генерация (МГЭС+ВЭУ+ДЭС+разряд АКБ) по часам;
        — charge_arrays: список массивов зарядных компонент (кВт), неотрицательные;
                         мы рисуем их вниз, поэтому масштаб нужен по абсолютной величине;
        — dump: массив сброса/балласта (кВт), неотрицательный (тоже рисуется вниз).

        Цель:
        — подобрать адекватные пределы y_min/y_max и шаг сетки (step),
          чтобы:
          • график не "прилипал" к границам;
          • сетка выглядела читаемо при разных мощностях (100 кВт vs 10 МВт);
          • диапазон симметричным быть не обязан — верх и низ выбираем отдельно.
        """

        # -------------------------
        # 1) Оцениваем верхний максимум
        # -------------------------
        # max_positive должен учитывать:
        # — load (линия нагрузки может быть выше суммарной генерации, если есть unserved);
        # — bottom (верх stacked bars).
        max_positive = max(
            bottom.max() if len(bottom) > 0 else 0,
            load.max() if len(load) > 0 else 0
        )

        # -------------------------
        # 2) Оцениваем нижний максимум (по модулю)
        # -------------------------
        # max_negative — максимум величины, которую мы рисуем вниз:
        # заряд АКБ и dump.
        max_negative = 0
        for arr in charge_arrays:
            if len(arr) > 0:
                max_negative = max(max_negative, arr.max())
        if len(dump) > 0:
            max_negative = max(max_negative, dump.max())

        # -------------------------
        # 3) Выбираем "разумный" шаг сетки
        # -------------------------
        # max_value — общий масштаб по модулю, чтобы выбрать step.
        max_value = max(max_positive, max_negative)

        # Эвристика:
        # — до 100 кВт: шаг 25;
        # — до 500 кВт: шаг 50;
        # — до 1 МВт: шаг 100;
        # — до 5 МВт: шаг 500;
        # — иначе: шаг 1000.
        # Дальше можно будет расширить (например 2_000/5_000/10_000 кВт).
        if max_value <= 100:
            step = 25  # Маленькие значения
        elif max_value <= 500:
            step = 50
        elif max_value <= 1000:
            step = 100
        elif max_value <= 5000:
            step = 500
        else:
            step = 1000

        # -------------------------
        # 4) Вычисляем y_max / y_min с небольшим запасом
        # -------------------------
        # y_max: округляем вверх до шага и добавляем еще один шаг "воздуха".
        if max_positive > 0:
            y_max = math.ceil(max_positive / step) * step + step
        else:
            y_max = step * 2

        # y_min: аналогично вниз, но отрицательный.
        if max_negative > 0:
            y_min = - (math.ceil(max_negative / step) * step + step)
        else:
            y_min = -step

        # Применяем пределы.
        self.ax.set_ylim(y_min, y_max)

        # -------------------------
        # 5) Декорации: нулевая линия и сетка
        # -------------------------
        # Нулевая линия: разделяет "генерацию/разряд" и "заряд/сброс".
        self.ax.axhline(y=0, color='#7f8c8d', linestyle='-', alpha=0.7, linewidth=1)

        # Major/minor сетка:
        # — major: шаг step;
        # — minor: step/2 для мягкого визуального ориентирования.
        self.ax.yaxis.set_major_locator(plt.MultipleLocator(step))
        self.ax.yaxis.set_minor_locator(plt.MultipleLocator(step / 2))
        self.ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.grid(True, which='minor', alpha=0.1, linestyle=':', linewidth=0.3)

        # Подпись на графике, чтобы пользователь видел выбранный step.
        self.ax.text(0.02, 0.98, f'Шаг сетки: {step} кВт',
                     transform=self.ax.transAxes, fontsize=8,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def update_chart(self, optimization_result, day=1):
        """
        Обновить график под выбранные сутки.

        Параметр day трактуется как номер суток (1..N), потому что start_hour = (day-1)*24.
        Текущий title пишет "типовой день месяца {day:02d}" — это формулировка,
        но по факту это именно "сутки №day" из оси времени (индекса schedule).
        """
        if optimization_result is None or not hasattr(optimization_result, 'schedule'):
            # Если результата нет — рисуем заглушку.
            self.draw_no_data_message()
            return

        schedule = optimization_result.schedule
        start_hour = (day - 1) * 24
        title = f'Покрытие нагрузки (типовой день месяца {day:02d})'
        self.plot_day(schedule, start_hour, title)

    def plot_day(self, schedule, start_idx, title_str):
        """
        Нарисовать 24 часа начиная с start_idx.

        Шаги:
        1) взять срез schedule[ start_idx : start_idx+24 ];
        2) достать ряды нагрузки/генерации/AKB/dump;
        3) определить агрегатные колонки ДГУ (если они есть);
        4) нарисовать stacked bars вверх (генерация);
        5) нарисовать stacked bars вниз (заряд/сброс);
        6) настроить стиль и автомасштаб.
        """
        self.ax.clear()

        hours_in_day = 24
        end_hour = min(start_idx + hours_in_day, len(schedule))

        # Срез данных на сутки
        day_data = schedule.iloc[start_idx:end_hour]

        # Если срез пустой — нечего рисовать
        if len(day_data) == 0:
            self.draw_no_data_message()
            return

        # Ось X: 1..24 (человеко-ориентировано)
        hours = list(range(1, len(day_data) + 1))

        # -------------------------
        # 1) Извлекаем базовые серии
        # -------------------------
        # .get(..., Series([0]*N)) — устойчиво к отсутствию колонок.
        load = day_data.get('load', pd.Series([0] * len(day_data))).values
        hydro = day_data.get('hydro', pd.Series([0] * len(day_data))).values
        wind = day_data.get('wind', pd.Series([0] * len(day_data))).values

        # diesel_total (суммарная ДЭС) — если нет агрегатов.
        # Если агрегатные колонки есть — будем рисовать их отдельно.
        diesel_total = day_data.get('diesel', pd.Series([0] * len(day_data))).values

        # Заряд/разряд берем через общую функцию, чтобы:
        # — не двойно учитывать синонимы колонок;
        # — предпочитать агрегированные charge_power/discharge_power, если они есть;
        # — отдавать metadata, по которому UI может красиво подписать источники.
        charge_discharge = compute_charge_discharge(day_data)
        battery_charge = charge_discharge['charge_kw']       # неотрицательный ряд кВт
        battery_discharge = charge_discharge['discharge_kw'] # неотрицательный ряд кВт

        charge_metadata = charge_discharge['metadata']
        # charge_components — словарь {имя_колонки: np.ndarray} только если charge_source="detailed"
        # и удалось выбрать компоненты.
        charge_components = charge_metadata.get('charge_components', {})

        dump = day_data.get('dump', pd.Series([0] * len(day_data))).values

        # -------------------------
        # 2) Находим колонки по агрегатам ДЭС
        # -------------------------
        # Логика: все, что не входит в known_columns — считаем агрегатами ДЭС.
        # Это нужно потому, что greedy пишет в schedule_row пары:
        #   "Plant - DGU 1": power, "Plant - DGU 2": power, ...
        known_columns = {
            'hour', 'load', 'hydro', 'wind', 'renewable_to_load',
            'battery_discharge', 'battery_discharge_to_load', 'battery_charge_from_renewable',
            'battery_charge_from_diesel', 'battery_charge_from_RES', 'battery_charge_from_DIESEL',
            'diesel', 'dump', 'unserved', 'diesel_excess', 'soc', 'soc_target',
            'soc_kwh', 'charge_power', 'discharge_power'
        }
        diesel_unit_cols = [col for col in day_data.columns if col not in known_columns]

        # -------------------------
        # 3) Цветовая схема (фиксированные цвета для последовательности)
        # -------------------------
        # Здесь выбран набор, который хорошо различим и на темной, и на светлой теме.
        colors = {
            'hydro': '#1f77b4',
            'wind': '#2ca02c',
            'diesel': '#b71c1c',
            'battery_discharge': '#8e44ad',
            'battery_charge_renewable': '#00bcd4',
            'battery_charge_diesel': '#f1c40f',
            'dump': '#7f8c8d',
            'load': '#000000'
        }

        # -------------------------
        # 4) Положительные компоненты (stacked bars вверх)
        # -------------------------
        # bottom — текущая "высота" уже нарисованных столбцов.
        # Для stacked bar: каждый следующий рисуется с bottom=bottom, затем добавляется.
        bottom = np.zeros(len(day_data))

        # МГЭС
        if hydro.sum() > 0.1:
            self.ax.bar(hours, hydro, bottom=bottom, label='МГЭС',
                        color=colors['hydro'], alpha=0.9, width=0.8)
            bottom += hydro

        # ВЭУ
        if wind.sum() > 0.1:
            self.ax.bar(hours, wind, bottom=bottom, label='ВЭУ',
                        color=colors['wind'], alpha=0.9, width=0.8)
            bottom += wind

        # ДЭС: либо по агрегатам, либо одной суммой
        if diesel_unit_cols:
            # Палитра красных оттенков для агрегатов, чтобы:
            # — визуально было понятно, что это одна "группа" (ДЭС),
            # — но агрегаты различались (разные оттенки).
            red_palette = ['#7f0000', '#a30000', '#c21807', '#d32f2f', '#e53935', '#f44336']
            diesel_palette = [red_palette[i % len(red_palette)] for i in range(len(diesel_unit_cols))]

            for idx, col in enumerate(sorted(diesel_unit_cols)):
                unit_values = day_data.get(col, pd.Series([0] * len(day_data))).values
                # Порог 0.1 — чтобы не захламлять легенду "почти нулевыми" столбцами.
                if unit_values.sum() > 0.1:
                    color = diesel_palette[idx] if len(diesel_unit_cols) > 1 else colors['diesel']
                    self.ax.bar(hours, unit_values, bottom=bottom, label=col,
                                color=color, alpha=0.9, width=0.8)
                    bottom += unit_values
        elif diesel_total.sum() > 0.1:
            # Если агрегатных колонок нет — рисуем суммарную ДЭС.
            self.ax.bar(hours, diesel_total, bottom=bottom, label='ДЭС',
                        color=colors['diesel'], alpha=0.9, width=0.8)
            bottom += diesel_total

        # Разряд АКБ (вверх, потому что он покрывает нагрузку)
        if battery_discharge.sum() > 0.1:
            self.ax.bar(hours, battery_discharge, bottom=bottom, label='АКБ → нагрузка',
                        color=colors['battery_discharge'], alpha=0.9, width=0.8)
            bottom += battery_discharge

        # Линия нагрузки:
        # — рисуем поверх (zorder=10), чтобы ее не перекрывали столбцы.
        self.ax.plot(hours, load, color=colors['load'], linewidth=3,
                     label='Нагрузка', zorder=10)

        # -------------------------
        # 5) Отрицательные компоненты (stacked bars вниз)
        # -------------------------
        # charge_bottom — аналог bottom, но уходит в отрицательную область.
        # Мы рисуем отрицательные значения (минус серия), а затем уменьшаем charge_bottom.
        charge_bottom = np.zeros(len(day_data))

        # Список зарядных компонент для автомасштаба (по модулю).
        # Нужен, потому что ось Y должна учитывать "насколько вниз" рисовать.
        charge_arrays_for_scale = []

        # Вариант 1: заряд восстановлен "detailed" и есть конкретные компоненты.
        # Тогда показываем, откуда зарядился АКБ: от ВИЭ или от ДЭС.
        if charge_metadata.get('charge_source') == 'detailed' and charge_components:
            renewable_cols = ['battery_charge_from_renewable', 'battery_charge_from_RES']
            diesel_cols = ['battery_charge_from_diesel', 'battery_charge_from_DIESEL']

            # Внутри compute_charge_discharge уже выбран "один" столбец-источник из синонимов,
            # но здесь еще раз аккуратно выбираем первый доступный ключ.
            renewable_series = None
            for col in renewable_cols:
                if col in charge_components:
                    renewable_series = charge_components[col]
                    break

            diesel_series = None
            for col in diesel_cols:
                if col in charge_components:
                    diesel_series = charge_components[col]
                    break

            # Заряд от ВИЭ
            if renewable_series is not None and renewable_series.sum() > 0.1:
                self.ax.bar(hours, -renewable_series, bottom=charge_bottom,
                            label='АКБ (заряд от ВИЭ)', color=colors['battery_charge_renewable'],
                            alpha=0.8, width=0.8)
                charge_bottom -= renewable_series
                charge_arrays_for_scale.append(renewable_series)

            # Заряд от ДЭС
            if diesel_series is not None and diesel_series.sum() > 0.1:
                self.ax.bar(hours, -diesel_series, bottom=charge_bottom,
                            label='АКБ (заряд от ДЭС)', color=colors['battery_charge_diesel'],
                            alpha=0.8, width=0.8)
                charge_bottom -= diesel_series
                charge_arrays_for_scale.append(diesel_series)

            # Если по какой-то причине компоненты не отрисовались, но общий заряд есть —
            # рисуем его одной серией, чтобы не потерять информацию.
            if not charge_arrays_for_scale and battery_charge.sum() > 0.1:
                self.ax.bar(hours, -battery_charge, bottom=charge_bottom,
                            label='АКБ (заряд)', color=colors['battery_charge_renewable'],
                            alpha=0.8, width=0.8)
                charge_bottom -= battery_charge
                charge_arrays_for_scale.append(battery_charge)

        # Вариант 2: заряд агрегированный (charge_power) или компоненты не выделились.
        else:
            if battery_charge.sum() > 0.1:
                self.ax.bar(hours, -battery_charge, bottom=charge_bottom,
                            label='АКБ (заряд)', color=colors['battery_charge_renewable'],
                            alpha=0.8, width=0.8)
                charge_bottom -= battery_charge
                charge_arrays_for_scale.append(battery_charge)

        # Сброс/балласт (dump) рисуем вниз.
        # Это "лишняя" мощность, которую некуда деть (АКБ полна/ограничения по мощности и т.п.).
        if dump.sum() > 0.1:
            self.ax.bar(hours, -dump, bottom=charge_bottom,
                        label='Сброс/балласт', color=colors['dump'],
                        alpha=0.8, width=0.8)

        # -------------------------
        # 6) Оформление + автомасштаб
        # -------------------------
        self.setup_chart_appearance(title_str, hours, load)

        # Если по какой-то причине список для масштаба пуст — подстрахуемся.
        if not charge_arrays_for_scale:
            charge_arrays_for_scale = [battery_charge]

        self.auto_scale_axes(load, bottom, charge_arrays_for_scale, dump)

        # tight_layout с выделением места под легенду справа:
        # rect=[left, bottom, right, top] в долях фигуры.
        self.fig.tight_layout(rect=[0, 0, 0.75, 0.95])
        self.draw()

    def setup_chart_appearance(self, title_str, hours, load):
        """
        Визуальный стиль EnergyChart.

        Почему это вынесено отдельно:
        — чтобы plot_day не раздувался еще сильнее;
        — чтобы можно было переиспользовать стиль при другом типе графика.
        """
        # Заголовок и подписи осей
        self.ax.set_title(title_str,
                          fontsize=14, fontweight='bold', pad=20, color='#2c3e50')

        self.ax.set_xlabel('Час', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel('Мощность, кВт', fontsize=12, fontweight='bold', color='#2c3e50')

        # Легенда с выносом вправо:
        # bbox_to_anchor=(1.02, 1) — ставим легенду чуть правее оси.
        legend = self.ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=9,
            framealpha=0.95,
            edgecolor='#bdc3c7',
            facecolor='#ecf0f1'
        )
        legend.set_title('Компоненты:', prop={'size': 10, 'weight': 'bold'})

        # Сетка "по умолчанию" (основную сетку усиливает auto_scale_axes)
        self.ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
        self.ax.set_axisbelow(True)

        # Ось X:
        # — часы от 1 до N;
        # — метки через 2 часа, чтобы не было наложения текста.
        self.ax.set_xlim(0.5, len(hours) + 0.5)
        self.ax.set_xticks(range(1, len(hours) + 1, 2))
        self.ax.tick_params(axis='both', which='major', labelsize=10)

    def draw_no_data_message(self, message="Нет данных для отображения"):
        """
        Заглушка, когда данных нет.

        Важно: мы очищаем ось и выключаем ее, чтобы не оставалось "старых" графиков.
        """
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        self.ax.axis('off')
        self.draw()


class BatteryChart(FigureCanvas):
    """
    График АКБ за сутки: SoC (кВт*ч) и мощности заряда/разряда (кВт).

    Задача этого графика:
    — быстро проверять, "вяжется ли" SoC с зарядом/разрядом;
    — видеть нарушения границ SOC min/max;
    — видеть, были ли одновременные заряд/разряд (это отлавливается на уровне алгоритма,
      но график помогает визуально).
    """

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)

        # Две оси:
        # верхняя — SoC (энергия);
        # нижняя — мощность (заряд/разряд).
        self.ax_soc = self.fig.add_subplot(211)
        self.ax_power = self.fig.add_subplot(212, sharex=self.ax_soc)

        # Отступ справа под легенду
        self.fig.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.1, hspace=0.25)

    @staticmethod
    def _clamp(value, min_val=None, max_val=None):
        """
        Утилита "зажать" число в диапазоне.
        Сейчас в данном файле напрямую не используется, но часто удобна при обработке входов.
        """
        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value

    def _compute_profile(self, day_data: pd.DataFrame, battery_meta: dict, day_start_soc: float = None, soc_start_source: str = ""):
        """
        Собирает все, что нужно для отрисовки АКБ за сутки, и возвращает словарь profile.

        Почему здесь много логики:
        — UI должен одинаково работать, даже если алгоритм:
          a) записал готовый столбец soc/soc_kwh;
          b) НЕ записал soc, но записал charge/discharge;
          c) записал charge/discharge в разных вариантах (aggregated/detailed).

        Основные шаги:
        1) определить стартовый SoC (day_start_soc);
        2) посчитать energy balance метрики через compute_battery_energy_balance();
        3) извлечь charge/discharge через compute_charge_discharge();
        4) взять soc_series из данных или "восстановить" его из charge/discharge;
        5) проверить выход за SOC min/max;
        6) подготовить компактный profile для plot_battery_day().
        """
        if day_data is None or day_data.empty:
            raise ValueError("Нет данных для выбранной даты")

        hours = list(range(1, len(day_data) + 1))

        # capacity_total — сколько кВт*ч суммарно у выбранной батареи (agg battery xN).
        capacity = battery_meta.get('capacity_total', 0) or 0

        # Границы могут быть:
        # — frac (0..1) в battery_meta["soc_min"/"soc_max"];
        # — kWh в battery_meta["soc_min_kwh"/"soc_max_kwh"].
        soc_min_frac = battery_meta.get('soc_min')
        soc_max_frac = battery_meta.get('soc_max')
        soc_min_kwh = battery_meta.get('soc_min_kwh')
        soc_max_kwh = battery_meta.get('soc_max_kwh')

        # КПД:
        # В BatteryState ты уже приводишь round-trip -> eta_charge/eta_discharge (sqrt),
        # но в meta здесь могут прилететь любые значения.
        eta_charge = battery_meta.get('eta_charge', 1.0) or 1.0
        eta_discharge = battery_meta.get('eta_discharge', 1.0) or 1.0

        # 1) Стартовый SoC суток.
        # Логика приоритета:
        # — если извне явно передали day_start_soc — используем его;
        # — иначе берем battery_meta["soc_start_kwh"] (то, что алгоритм считает стартом).
        if day_start_soc is None:
            day_start_soc = battery_meta.get('soc_start_kwh')
        if day_start_soc is not None:
            day_start_soc = float(day_start_soc)

        # Источник старта:
        # — "initial"/"provided"/"fallback" и т.п. (для отладки и подписи).
        soc_start_source = soc_start_source or battery_meta.get('soc_start_source', '')

        # 2) Баланс энергии (SOC-side).
        # compute_battery_energy_balance считает ожидаемую дельту SoC через:
        #   sum(P_charge * eta_charge * dt) - sum(P_discharge / eta_discharge * dt)
        # и сравнивает с фактической delta SoC (soc_end - soc_start).
        balance = compute_battery_energy_balance(
            day_data,
            {**battery_meta, 'soc_min_kwh': soc_min_kwh, 'soc_max_kwh': soc_max_kwh},
            soc_start_override=day_start_soc,
        )

        # 3) Заряд/разряд в кВт.
        # Здесь снова принцип: только compute_charge_discharge() определяет,
        # какие колонки использовать (чтобы не было двойного учета).
        charge_discharge = compute_charge_discharge(day_data)
        charge_power = charge_discharge['charge_kw']       # заряд, кВт, неотрицательный
        discharge_power = charge_discharge['discharge_kw'] # разряд, кВт, неотрицательный

        # 4) soc_series:
        # — если алгоритм сформировал колонку 'soc', используем ее;
        # — иначе восстанавливаем SoC интегрированием по времени из charge/discharge.
        soc_series = day_data['soc'].values if 'soc' in day_data.columns else None
        if soc_series is None:
            # Восстановление SoC:
            # — начинаем с balance['soc_start_kwh'];
            # — на каждом шаге delta = eta_charge*P_charge - P_discharge/eta_discharge.
            #
            # ВАЖНО:
            # Здесь предполагается dt=1 час (что соответствует твоим данным 8760/час).
            # Если когда-то перейдешь на другой dt, надо будет умножать delta на dt_hours.
            soc_values = [balance['soc_start_kwh']]
            for ch, dis in zip(charge_power, discharge_power):
                delta = eta_charge * ch - dis / max(eta_discharge, 1e-9)
                soc_values.append(soc_values[-1] + delta)
            soc_series = np.array(soc_values[1:])

            if soc_start_source == "":
                soc_start_source = "fallback"
        else:
            if soc_start_source == "":
                soc_start_source = "provided"

        # 5) Предупреждения/проверки.
        warnings = list(balance['warnings'])

        # Если границы в кВт*ч не заданы, но есть доли и емкость — считаем в кВт*ч.
        if soc_min_kwh is None and soc_min_frac is not None:
            soc_min_kwh = soc_min_frac * capacity
        if soc_max_kwh is None and soc_max_frac is not None:
            soc_max_kwh = soc_max_frac * capacity

        # Проверка "выхода" SoC за границы.
        # Порог 1e-3 — чтобы не ловить микродрейф float.
        if soc_min_kwh is not None and soc_max_kwh is not None:
            if (soc_series < soc_min_kwh - 1e-3).any() or (soc_series > soc_max_kwh + 1e-3).any():
                warnings.append("SOC violation")

        # Итоговый profile: все, что нужно отрисовать + метрики баланса.
        return {
            'hours': hours,
            'charge_power': charge_power,
            'discharge_power': discharge_power,
            'soc_series': soc_series,
            'soc_start': balance['soc_start_kwh'],
            'soc_end': balance['soc_end_kwh'],
            'soc_min_kwh': soc_min_kwh,
            'soc_max_kwh': soc_max_kwh,
            'total_charge_kwh': round(balance['total_charge_kwh'], 6),
            'total_discharge_kwh': round(balance['total_discharge_kwh'], 6),
            'delta_soc_kwh': round(balance['delta_soc_kwh'], 6),
            'eta_used': (eta_charge + eta_discharge) / 2 if eta_charge != 1 or eta_discharge != 1 else None,
            'warnings': warnings,
            'soc_start_source': soc_start_source,
        }

    def plot_battery_day(self, day_data: pd.DataFrame, date_label: str, battery_meta: dict, day_start_soc: float = None, soc_start_source: str = ""):
        """
        Отрисовать профиль АКБ за сутки.

        Визуальная логика:
        — SoC рисуем линией;
        — SOC min/max рисуем пунктирными горизонтальными линиями;
        — мощности: discharge вверх, charge вниз (с минусом для charge).
        """
        profile = self._compute_profile(day_data, battery_meta, day_start_soc=day_start_soc, soc_start_source=soc_start_source)

        # Очищаем оси перед новой отрисовкой
        self.ax_soc.clear()
        self.ax_power.clear()

        hours = profile['hours']
        soc_series = profile['soc_series']
        charge_power = profile['charge_power']
        discharge_power = profile['discharge_power']

        # Верхний график: SoC
        self.ax_soc.plot(hours, soc_series, label='SoC', color='#2c3e50', linewidth=2)

        # Границы SoC (если известны)
        if profile['soc_min_kwh'] is not None:
            self.ax_soc.axhline(profile['soc_min_kwh'], color='#c0392b', linestyle='--', label='SOC min')
        if profile['soc_max_kwh'] is not None:
            self.ax_soc.axhline(profile['soc_max_kwh'], color='#27ae60', linestyle='--', label='SOC max')

        self.ax_soc.set_ylabel('Энергия, кВт·ч')
        self.ax_soc.set_title(f'Состояние заряда АКБ — {date_label}', fontsize=12, fontweight='bold')
        self.ax_soc.grid(True, which='major', alpha=0.3)

        # Легенду SoC выносим вправо, чтобы не закрывать график
        self.ax_soc.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Нижний график: мощности
        # discharge — вверх; charge — вниз (поэтому -charge_power).
        self.ax_power.bar(hours, discharge_power, width=0.8, label='P_discharge', color='#d35400')
        self.ax_power.bar(hours, -charge_power, width=0.8, label='P_charge', color='#2980b9')
        self.ax_power.axhline(0, color='#7f8c8d', linewidth=1)

        self.ax_power.set_ylabel('Мощность, кВт')
        self.ax_power.set_xlabel('Час')
        self.ax_power.set_title('Заряд/разряд АКБ', fontsize=11)
        self.ax_power.grid(True, which='major', alpha=0.3)
        self.ax_power.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

        # Ось X: метки через 2 часа (чтобы не лепились)
        self.ax_power.set_xlim(0.5, len(hours) + 0.5)
        self.ax_power.set_xticks(range(1, len(hours) + 1, 2))

        # Учет места справа под легенды
        self.fig.tight_layout(rect=[0, 0, 0.78, 0.95])
        self.draw()

        return profile

    def draw_no_data_message(self, message="Нет данных для отображения"):
        """
        Заглушка, когда для АКБ нечего показать.
        Очищаем обе оси, выключаем их и пишем сообщение.
        """
        self.ax_soc.clear()
        self.ax_power.clear()
        self.ax_soc.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
        self.ax_power.axis('off')
        self.ax_soc.axis('off')
        self.draw()
