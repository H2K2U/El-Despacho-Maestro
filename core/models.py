# File: core/models.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np


@dataclass
class DieselUnit:
    """
    Модель одной дизель-генераторной установки (ДГУ / ДЭУ как агрегат).

    Поля:
    • name — имя агрегата (для вывода/таблиц/колонок в schedule);
    • nominal_power — номинальная электрическая мощность агрегата, кВт;
    • efficiency — КПД (сейчас в расчетах топлива напрямую не используется, но может быть полезен);
    • fuel_curve — табличная кривая расхода топлива:
      ожидается DataFrame с колонками:
      - 'load' — нагрузка в кВт (абсолютная, не доля);
      - 'consumption' — расход топлива (в чем? вероятнее л/ч или г/кВт*ч * кВт, но это надо унифицировать).
    """

    name: str
    nominal_power: float  # кВт
    efficiency: float  # КПД
    fuel_curve: pd.DataFrame  # Кривая расхода топлива (нагрузка -> расход)

    def fuel_consumption(self, load_percentage: float) -> float:
        """
        Расчет расхода топлива при заданной относительной нагрузке агрегата.

        Параметры:
        • load_percentage — доля загрузки агрегата в диапазоне [0..1],
          например 0.5 означает "половина номинала".

        Что делает:
        1) Если fuel_curve отсутствует → возвращает 0 (то есть "топливо не считаем").
        2) Переводит относительную нагрузку в абсолютную:
           load = load_percentage * nominal_power.
        3) Интерполирует расход по кривой fuel_curve:
           np.interp(x, xp, fp) строит линейную интерполяцию.

        Критический нюанс:
        • Если load_percentage выходит за рамки [0..1], np.interp экстраполирует по краям
          (на самом деле возвращает крайние значения). Лучше бы явно clamp-ить в [0..1].
        • Нет проверки наличия колонок 'load' и 'consumption' — при неверном формате упадет KeyError.
        """
        if self.fuel_curve is None:
            return 0
        # load — абсолютная мощность (кВт) по доле загрузки
        load = load_percentage * self.nominal_power
        return np.interp(load, self.fuel_curve['load'], self.fuel_curve['consumption'])


@dataclass
class DieselPlant:
    """
    Модель дизельной электростанции (ДЭС) как набора агрегатов (DieselUnit).

    Поля:
    • name — имя ДЭС (используется, например, чтобы делать колонки вида "ДЭС-1 - ДГУ 1");
    • diesel_units — список агрегатов.

    Эта модель удобна тем, что:
    • можно считать суммарную установленную мощность;
    • можно подбирать набор агрегатов для покрытия требуемой нагрузки.
    """

    name: str
    diesel_units: List[DieselUnit]

    @property
    def total_capacity(self) -> float:
        """
        Суммарная установленная мощность ДЭС (кВт) = сумма номиналов всех агрегатов.
        """
        return sum(unit.nominal_power for unit in self.diesel_units)

    def get_optimal_units(self, required_power: float, min_load: float = 0.4) -> List[DieselUnit]:
        """
        Упрощенный подбор набора агрегатов для покрытия required_power.

        Параметры:
        • required_power — требуемая мощность (кВт), которую надо обеспечить ДЭС;
        • min_load — минимальная допустимая загрузка агрегата (доля от номинала),
          например 0.4 означает "не включаем агрегат, если ожидаемая нагрузка меньше 40% номинала".

        Алгоритм (жадный, но НЕ тот же, что в greedy_optimization):
        1) Сортируем агрегаты по nominal_power по возрастанию.
        2) Идем от малых к большим, добавляя агрегаты,
           пока remaining_power > 0.
        3) Добавляем агрегат только если remaining_power >= unit.nominal_power * min_load.
           То есть если нагрузка "достаточно большая", чтобы агрегат имел смысл запускать.

        Важно:
        • Здесь remaining_power уменьшается на unit.nominal_power (а не на реально поданную мощность).
          Это грубая оценка: будто агрегат всегда закрывает полный номинал.
          Поэтому это скорее "быстрый подбор состава", а не реальная диспетчеризация.
        • В твоем greedy_optimization есть более аккуратная логика distribute_power_among_units,
          которая распределяет по Pmin/Pmax. Эта функция здесь — упрощение/альтернатива.
        """
        sorted_units = sorted(self.diesel_units, key=lambda x: x.nominal_power)
        selected_units = []
        remaining_power = required_power

        for unit in sorted_units:
            if remaining_power <= 0:
                break
            if unit.nominal_power * min_load <= remaining_power:
                selected_units.append(unit)
                remaining_power -= unit.nominal_power

        return selected_units


@dataclass
class HydroPlant:
    """
    Модель малой ГЭС (МГЭС/ГЭС в твоем контексте) со "встроенным" гидрографом.

    Поля:
    • name — название;
    • nominal_power — ограничение по установленной мощности (кВт);
    • efficiency — КПД (η) турбина+генератор (доля 0..1);
    • head — напор H (м), по умолчанию 50 (очень условно);
    • hydro_graph — словарь {месяц: расход Q}, где Q в м^3/с.

    Главное:
    • available_power(month) возвращает доступную мощность по формуле
      P = ρ g H Q η, ограниченную nominal_power.
    """

    name: str
    nominal_power: float  # кВт
    efficiency: float
    head: float = 50.0  # напор, м
    hydro_graph: dict = None  # месяц -> расход (м³/с)

    def __post_init__(self):
        """
        __post_init__ автоматически вызывается dataclass-ом сразу после __init__.

        Здесь:
        • гарантируем, что hydro_graph всегда хотя бы пустой dict,
          чтобы методы могли безопасно обращаться к self.hydro_graph.
        """
        if self.hydro_graph is None:
            self.hydro_graph = {}

    def available_power(self, month: int) -> float:
        """
        Доступная мощность МГЭС в заданный месяц.

        Параметры:
        • month — номер месяца 1..12.

        Логика:
        1) Если hydro_graph задан и содержит month:
           • flow = Q (м^3/с)
           • power = 1000 * 9.81 * head * flow * efficiency / 1000
             То есть (ρ=1000 кг/м^3) * g * H * Q * η, затем /1000 для перевода Вт→кВт.
           • возвращаем min(power, nominal_power) — физическое ограничение установленной мощности.
        2) Если данных по месяцу нет:
           • возвращаем 50% от номинала (очень грубый fallback).
        """
        if self.hydro_graph and month in self.hydro_graph:
            flow = self.hydro_graph[month]  # м³/с
            power = 1000 * 9.81 * self.head * flow * self.efficiency / 1000  # кВт
            return min(power, self.nominal_power)
        else:
            return self.nominal_power * 0.5

    def set_hydro_graph(self, hydro_graph: dict):
        """
        Установка/обновление гидрографа.

        Параметры:
        • hydro_graph — словарь {месяц: расход}.
        """
        self.hydro_graph = hydro_graph


@dataclass
class WindTurbine:
    """
    Модель ветроустановки (ВЭУ).

    Поля:
    • name — название/модель;
    • nominal_power — номинальная мощность (кВт);
    • power_curve — DataFrame с кривой мощности:
      ожидаются колонки:
      - 'wind_speed' (м/с)
      - 'power' (кВт)
    • height — высота оси (м) для пересчета скорости ветра к высоте ВЭУ.
    • cut_in_speed — скорость включения (м/с);
    • rated_speed — скорость, при которой достигается nominal_power (м/с);
    • cut_out_speed — скорость отключения (м/с).

    Основная функция:
    • available_power(wind_speed) → мощность на данном ветре.
    """

    name: str
    nominal_power: float  # кВт
    power_curve: pd.DataFrame  # Кривая мощности (скорость ветра -> мощность)
    height: float  # Высота оси, м
    cut_in_speed: float = 3.0  # Скорость включения, м/с
    rated_speed: float = 15.0  # Номинальная скорость, м/с
    cut_out_speed: float = 25.0  # Скорость выключения, м/с

    def available_power(self, wind_speed: float, measurement_height: float = 10) -> float:
        """
        Расчет доступной мощности ВЭУ при скорости ветра wind_speed.

        Параметры:
        • wind_speed — измеренная скорость ветра (м/с) на высоте measurement_height;
        • measurement_height — высота измерения (м), по умолчанию 10 м (типовая метеостанция).

        Шаги:
        1) Если power_curve отсутствует/пустая → 0.
        2) Если wind_speed вне диапазона [cut_in_speed, cut_out_speed] → 0.
        3) Пересчитываем скорость ветра на высоту ВЭУ:
           • используется степенной закон (Hellmann exponent):
             v(z2) = v(z1) * (z2/z1)^alpha, alpha=0.14 (открытая местность).
           • это приближение; реальный alpha зависит от шероховатости, стабильности атмосферы и т.д.
        4) Пытаемся интерполировать по power_curve:
           np.interp(wind_speed, curve['wind_speed'], curve['power'])
        5) Если вдруг интерполяция "не удалась" (например, где-то NaN/не те типы):
           fallback-логика:
           • ниже cut-in / выше cut-out → 0
           • выше rated_speed → nominal_power
           • иначе линейный рост от cut-in до rated.

        Нюанс:
        • В ветке try/except "не удалась" может скрыть реальные ошибки данных.
          Иногда лучше ловить конкретные исключения и логировать.
        """
        if self.power_curve is None or self.power_curve.empty:
            return 0

        if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
            return 0

        # Пересчет скорости на высоту оси ВЭУ
        if measurement_height != self.height and measurement_height > 0:
            alpha = 0.14
            wind_speed = wind_speed * (self.height / measurement_height) ** alpha

        try:
            return np.interp(
                wind_speed,
                self.power_curve['wind_speed'],
                self.power_curve['power']
            )
        except:
            if wind_speed < self.cut_in_speed:
                return 0
            elif wind_speed > self.cut_out_speed:
                return 0
            elif wind_speed >= self.rated_speed:
                return self.nominal_power
            else:
                return self.nominal_power * (wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)


@dataclass
class Battery:
    """
    Модель аккумуляторной батареи (АКБ) в простом виде.

    Поля:
    • name — название;
    • capacity — емкость (кВт*ч);
    • max_charge_power — макс. мощность заряда (кВт);
    • max_discharge_power — макс. мощность разряда (кВт);
    • efficiency — КПД (в этой модели трактуется как "одинаковый на заряд и разряд");
    • soc_min / soc_max — ограничения SOC (обычно доли 0..1).

    ВАЖНО про совместимость:
    • У тебя есть отдельный "правильный" класс BatteryState, который работает в kWh и kW
      и аккуратно учитывает eta_charge/eta_discharge и ограничения по SOC.
    • Этот Battery — скорее спецификация + простой внутренний SOC.
      В greedy_optimization ты работаешь через BatteryState.from_battery(selected_battery),
      то есть фактически используешь Battery как "паспорт" батареи.
    """

    name: str
    capacity: float  # кВт*ч
    max_charge_power: float  # кВт
    max_discharge_power: float  # кВт
    efficiency: float = 0.95  # КПД заряда/разряда
    soc_min: float = 0.1  # Минимальный SOC
    soc_max: float = 0.9  # Максимальный SOC

    def __post_init__(self):
        """
        Инициализация динамического состояния батареи.

        • current_soc — текущий SOC (доля), стартуем с середины диапазона.
        • _last_action — запоминаем последнее действие ('charge'/'discharge'/None).
          Сейчас это поле нигде явно не используется в алгоритмах,
          но может пригодиться, например, чтобы штрафовать за частые переключения.
        """
        self.current_soc = (self.soc_min + self.soc_max) / 2.0  # начальный заряд 50%
        self._last_action = None  # 'charge', 'discharge', или None

    @property
    def eta_charge(self) -> float:
        """
        КПД заряда для совместимости с кодом оптимизации/метрик.

        Сейчас возвращает self.efficiency.
        Нюанс:
        • В BatteryState.from_battery efficiency трактуется как round-trip,
          и при необходимости делится на sqrt для получения one-way КПД.
        • Здесь же eta_charge=eta_discharge=efficiency напрямую.
          Это может создавать нестыковки, если одно и то же поле используется по-разному.
        """
        return self.efficiency

    @property
    def eta_discharge(self) -> float:
        """КПД разряда (аналогично eta_charge)."""
        return self.efficiency

    def can_charge(self):
        """
        Может ли батарея заряжаться по ограничению SOC.

        Возвращает True, если current_soc < soc_max - 0.001.
        • 0.001 — "мертвая зона", чтобы не дрожать на границе из-за float.
        """
        return self.current_soc < self.soc_max - 0.001

    def can_discharge(self):
        """
        Может ли батарея разряжаться по ограничению SOC.

        True, если current_soc > soc_min + 0.001.
        """
        return self.current_soc > self.soc_min + 0.001

    def charge(self, power_kw: float, duration_h: float = 1.0) -> float:
        """
        Заряд АКБ с учетом КПД.

        Параметры:
        • power_kw — запрошенная мощность заряда (кВт);
        • duration_h — длительность шага (часы), обычно 1 час.

        Возвращает:
        • actual_power — фактическая принимаемая мощность заряда (кВт),
          которая может быть меньше из-за:
          - лимита max_charge_power,
          - недостатка "места" до soc_max.

        Что происходит внутри:
        1) Если нельзя заряжаться (SOC уже почти на soc_max) → 0.
        2) max_power ограничивается двумя вещами:
           • max_charge_power;
           • оставшаяся емкость до soc_max (в кВт*ч), переведенная в кВт с учетом КПД:
             (soc_max - current_soc) * capacity / efficiency.
        3) actual_power = min(power_kw, max_power)
        4) Энергия, которая реально добавится в батарею:
           energy_stored = actual_power * duration_h * efficiency
        5) SOC увеличивается на energy_stored / capacity.

        Нюанс:
        • Эта модель не предотвращает одновременный заряд/разряд (это делает BatteryState.request).
        • Здесь efficiency стоит в знаменателе в max_power и в числителе в energy_stored —
          общая логика "на SOC стороне" выглядит правдоподобно, но должна быть строго согласована
          с BatteryState и метриками compute_battery_energy_balance.
        """
        if not self.can_charge():
            return 0.0

        max_power = min(
            self.max_charge_power,
            (self.soc_max - self.current_soc) * self.capacity / self.efficiency
        )
        actual_power = min(power_kw, max_power)

        if actual_power > 0:
            energy_stored = actual_power * duration_h * self.efficiency
            self.current_soc += energy_stored / self.capacity
            self._last_action = 'charge'

        return actual_power

    def discharge(self, power_kw: float, duration_h: float = 1.0) -> float:
        """
        Разряд АКБ с учетом КПД.

        Параметры:
        • power_kw — запрошенная мощность разряда (кВт) "на нагрузку";
        • duration_h — длительность шага (ч).

        Возвращает:
        • actual_power — фактическая отдаваемая мощность разряда (кВт), ограниченная:
          - max_discharge_power,
          - доступной энергией до soc_min.

        Логика:
        1) Если нельзя разряжаться (SOC близко к soc_min) → 0.
        2) max_power ограничивается:
           • max_discharge_power;
           • доступной энергией:
             (current_soc - soc_min) * capacity * efficiency
           (в этой формуле efficiency в числителе — то есть учитываются потери).
        3) actual_power = min(power_kw, max_power)
        4) Энергия, которая "спишется" с SOC:
           energy_delivered = actual_power * duration_h / efficiency
           SOC уменьшается на energy_delivered / capacity

        Нюанс:
        • Здесь в max_power efficiency в числителе, а в energy_delivered — в знаменателе.
          Это тот же "SOC-side" подход, но важно, чтобы он совпадал с BatteryState.request,
          иначе в проекте будут расхождения SOC и графиков (ты как раз ловил это раньше).
        """
        if not self.can_discharge():
            return 0.0

        max_power = min(
            self.max_discharge_power,
            (self.current_soc - self.soc_min) * self.capacity * self.efficiency
        )
        actual_power = min(power_kw, max_power)

        if actual_power > 0:
            energy_delivered = actual_power * duration_h / self.efficiency
            self.current_soc -= energy_delivered / self.capacity
            self._last_action = 'discharge'

        return actual_power


@dataclass
class LoadProfile:
    """
    Модель графика нагрузки.

    Поле:
    • data — DataFrame, где индекс обычно DatetimeIndex,
      а колонка нагрузки — 'load' (кВт).

    Метод get_daily_load:
    • позволяет взять 24 часа по номеру дня.
    """

    data: pd.DataFrame  # Время -> Потребление

    def get_daily_load(self, day: int) -> pd.Series:
        """
        Получить нагрузку для конкретного дня (1..365), срезом по 24 часа.

        Параметры:
        • day — номер дня в году (1..365).

        Возвращает:
        • self.data.iloc[start_idx:end_idx] — срез DataFrame/Series на 24 строки.

        Нюанс:
        • Это работает корректно только если data упорядочена ровно по часам
          и начинается с начала года без пропусков.
          Если будут пропуски/смещения/не 8760 точек — "день" может быть неверным.
        """
        if day < 1 or day > 365:
            raise ValueError("День должен быть в диапазоне 1-365")

        start_idx = (day - 1) * 24
        end_idx = day * 24
        return self.data.iloc[start_idx:end_idx]


@dataclass
class OptimizationResult:
    """
    Результат оптимизации (унифицированная структура, которую возвращают greedy/de/wolf).

    Поля:
    • schedule — DataFrame расписания по времени (обычно по часам), где есть:
      - 'load', 'hydro', 'wind', 'diesel', 'dump', 'unserved', 'soc', ...
      и/или детальные колонки по ДГУ;
    • selected_equipment — словарь с выбранным составом (какие ВЭУ/АКБ/ДГУ/МГЭС выбраны);
    • total_fuel_consumption — суммарный расход топлива (единицы зависят от DieselUnit.fuel_curve);
    • renewable_energy_ratio — доля энергии, покрытой ВИЭ (гидро+ветер) от общей нагрузки;
    • cost_analysis — словарь под экономику (CAPEX/OPEX/NPV/LCOE и т.п.), пока пустой.

    Нюанс:
    • Типы selected_equipment и cost_analysis — просто Dict без структуры.
      Это удобно на раннем этапе, но потом лучше типизировать (TypedDict / dataclass),
      чтобы UI и отчеты не ломались от изменений ключей.
    """

    schedule: pd.DataFrame  # Расписание работы оборудования
    selected_equipment: Dict  # Подобранное оборудование
    total_fuel_consumption: float
    renewable_energy_ratio: float
    cost_analysis: Dict
