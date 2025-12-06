# File: core/algorithms/wolf.py
"""
Реализация оптимизации методом "серых волков" (Grey Wolf Optimizer, GWO) для подбора состава оборудования.

1) Роль алгоритма в проекте
   Это алгоритм ВЕРХНЕГО уровня (sizing/selection):
   • он подбирает ТИП ВЭУ и КОЛИЧЕСТВО ВЭУ;
   • подбирает ТИП АКБ и КОЛИЧЕСТВО АКБ;
   а для оценки "качества" каждого кандидата вызывает нижний уровень (dispatch) — greedy_optimization(...),
   который строит почасовой график выдачи/заряда/разряда/сброса/недоотпуска.

2) Оптимизационная модель в терминах "целевая функция/ограничения" (как реализовано здесь)

2.1) Переменные решения (что оптимизируем)
   Обозначим x как вектор решения; он кодируется в numpy-массиве position размерности dim:
   • если есть wind_turbines:
       x0 = индекс типа ВЭУ (целое после округления) in [0 .. N_wt-1]
       x1 = количество ВЭУ (целое после округления) in [0 .. n_wind_max]
   • если есть batteries:
       x2 = индекс типа АКБ (целое после округления) in [0 .. N_bat-1]
       x3 = количество АКБ (целое после округления) in [0 .. n_batt_max]

   Внутри GWO оптимизация идет по "непрерывным" position, но перед оценкой они:
   • ограничиваются (clamp) по границам;
   • округляются (round) до целых индексов/количеств.

2.2) Целевая функция (скалярная)
   Здесь fitness — это штрафная функция "чем меньше, тем лучше":

   min  F(x) = 1e9 * E_unserved(x)
               + Fuel(x)
               + 5.0 * E_dump(x)
               + 1e3 * H_diesel(x)
               + CAPEX_proxy(x)

   где:
   • E_unserved — энергия недоотпуска (кВт*ч) за рассматриваемый горизонт (fast/full);
   • Fuel — общий расход топлива (из result.total_fuel_consumption, иначе прокси по дизельной энергии);
   • E_dump — энергия сброса (dump/ballast/curtailment, суммируется как кВт или кВт*ч при dt=1);
   • H_diesel — количество часов работы дизеля (сколько часов дизельная мощность > 0);
   • CAPEX_proxy — суррогат капитальных затрат:
        WIND_CAPEX_WEIGHT * (n_wind * P_nom_wt[type])
      + BATTERY_CAPEX_WEIGHT * (n_batt * Capacity_bat[type])

   Главный приоритет — "запретить" недоотпуск (множитель 1e9), затем топливо и остальные штрафы.

   ВАЖНО:
   • Это ОДНОКРИТЕРИАЛЬНАЯ постановка (все цели сведены в один fitness весами).
   • Весовые коэффициенты не нормированы, это эвристика.

2.3) Ограничения
   Явные ограничения на x:
   • индексы типов: целые, ограничены размером каталогов (0..N-1);
   • количества: целые, ограничены сверху n_wind_max, n_batt_max (дополнительно ограничены 200).

   Неявные (внутри greedy_optimization):
   • почасовой баланс мощности, ограничения ДГУ по Pmin/Pmax, ограничения АКБ по SOC и мощности,
     ограничения ВИЭ по доступной мощности и т.д.

3) Ускорение: двухэтапная оценка FAST/FULL
   Чтобы не гонять greedy на 8760 часов для каждого волка на каждой итерации:
   • FAST: выбирается "худшее окно" длиной 168 часов (неделя) по максимальной сумме нагрузки;
     оптимизация (поиск) ведется в основном по этому окну;
   • FULL: раз в full_interval итераций (и в конце) берутся top_k уникальных кандидатов,
     и они прогоняются на полном горизонте (весь load_profile).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.algorithms.greedy import greedy_optimization
from core.models import Battery, DieselPlant, HydroPlant, LoadProfile, OptimizationResult, WindTurbine


# Вес-прокси "капвложений" ВЭУ в fitness.
# Интерпретация: добавляем к fitness штраф пропорционально установленной мощности ВЭУ.
WIND_CAPEX_WEIGHT = 0.1

# Вес-прокси "капвложений" АКБ в fitness.
# Интерпретация: добавляем к fitness штраф пропорционально суммарной емкости АКБ.
BATTERY_CAPEX_WEIGHT = 0.05


def _clamp(value: float, lower: float, upper: float) -> float:
    """Ограничение value в диапазон [lower, upper]."""
    return max(lower, min(upper, value))


def _get_load_series(load_profile: LoadProfile) -> pd.Series:
    """
    Достает временной ряд нагрузки из load_profile.

    Ожидаемый формат: load_profile.data — DataFrame, где:
    • колонка "load" — предпочтительно;
    • иначе берем "первую числовую" колонку как запасной вариант.

    Возвращаем pd.Series:
    • пустой Series(dtype=float), если не смогли корректно извлечь данные.
    """
    if isinstance(load_profile.data, pd.DataFrame):
        if "load" in load_profile.data.columns:
            return load_profile.data["load"]
        # Запасной вариант: берем первую числовую колонку.
        numeric_cols = load_profile.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return load_profile.data[numeric_cols[0]]
    return pd.Series(dtype=float)


def _compute_limits(
    load_profile: LoadProfile,
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
) -> Tuple[int, int]:
    """
    Оценивает верхние границы количества ВЭУ и АКБ по пику нагрузки.

    p_peak — максимальная нагрузка (кВт).

    Дальше:
    • k_over — "перебор" по ветру: допускаем, что суммарная номинальная мощность ВЭУ может быть
      до k_over * p_peak (чтобы компенсировать непостоянство ветра);
    • k_batt — "перебор" по АКБ: допускаем, что суммарная разрядная мощность АКБ может быть
      до k_batt * p_peak.

    min_wind_power — минимальная номинальная мощность среди каталогов ВЭУ (кВт),
    min_batt_power — минимальная max_discharge_power среди каталогов АКБ (кВт),
    чтобы получить "наихудший" (наиболее большой) верхний предел по количеству.

    Возвращает:
    • n_wind_max — максимум количества ВЭУ (не более 200);
    • n_batt_max — максимум количества АКБ (не более 200).
    """
    load_series = _get_load_series(load_profile)
    p_peak = float(load_series.max()) if not load_series.empty else 0.0

    # Эвристические коэффициенты запаса.
    k_over = 2.0
    k_batt = 2.0

    # Минимальные мощности в каталогах (чтобы получить верхнюю границу по количеству).
    min_wind_power = min((wt.nominal_power for wt in wind_turbines), default=0.0)
    min_batt_power = min((bt.max_discharge_power for bt in batteries), default=0.0)

    # Верхние границы по количеству (с ограничением до 200).
    n_wind_max = int(np.ceil((k_over * p_peak) / min_wind_power)) if min_wind_power > 0 else 0
    n_batt_max = int(np.ceil((k_batt * p_peak) / min_batt_power)) if min_batt_power > 0 else 0

    return min(n_wind_max, 200), min(n_batt_max, 200)


def _decode_candidate(
    position: np.ndarray,
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    n_wind_max: int,
    n_batt_max: int,
) -> Tuple[int, int, int, int]:
    """
    Преобразует "непрерывную" позицию волка (position) в дискретное решение:
    (wind_idx, n_wind, batt_idx, n_batt).

    position содержит значения в диапазонах, заданных lower_bounds/upper_bounds_arr.
    Но индексы и количества здесь должны быть целыми, поэтому:
    • сначала clamp по границам;
    • затем round и int.

    Возвращает:
    • wind_idx — индекс выбранного типа ВЭУ в каталоге;
    • n_wind — количество ВЭУ;
    • batt_idx — индекс выбранного типа АКБ;
    • n_batt — количество АКБ.
    """
    idx = 0
    wind_idx = 0
    n_wind = 0
    batt_idx = 0
    n_batt = 0

    if wind_turbines:
        wind_idx = int(round(_clamp(position[idx], 0, len(wind_turbines) - 1)))
        idx += 1
        n_wind = int(round(_clamp(position[idx], 0, n_wind_max)))
        idx += 1

    if batteries:
        batt_idx = int(round(_clamp(position[idx], 0, len(batteries) - 1)))
        idx += 1
        n_batt = int(round(_clamp(position[idx], 0, n_batt_max)))

    return wind_idx, n_wind, batt_idx, n_batt


def _get_diesel_unit_names(diesel_plants: List[DieselPlant]) -> List[str]:
    """
    Генерирует список человекочитаемых имен дизельных агрегатов (ДГУ),
    чтобы потом корректно собирать дизельные колонки из schedule.

    Формат имени:
    • если у станции есть имя: "Имя станции - Имя агрегата"
    • иначе: "Имя агрегата"
    • если у агрегата нет имени: "ДГУ <номер>"
    """
    names: List[str] = []
    unit_counter = 1
    for plant in diesel_plants:
        for unit in plant.diesel_units:
            unit_name = unit.name or f"ДГУ {unit_counter}"
            full_name = f"{plant.name} - {unit_name}" if plant.name else unit_name
            names.append(full_name)
            unit_counter += 1
    return names


def _aggregate_wind(wind_turbines: List[WindTurbine], wind_idx: int, n_wind: int) -> List[WindTurbine]:
    """
    "Агрегирует" n_wind одинаковых ВЭУ в один эквивалентный объект WindTurbine.

    Зачем:
    • greedy_optimization ожидает список wind_turbines; но для оптимизации удобнее хранить
      решение как (тип, количество).
    • Мы создаем один объект с номинальной мощностью base.nominal_power * n_wind,
      а также масштабируем кривую мощности (power_curve["power"] *= n_wind), если она есть.

    Возвращает список из одного агрегированного WindTurbine, либо пустой список (если n_wind <= 0).
    """
    if not wind_turbines or n_wind <= 0:
        return []

    base = wind_turbines[wind_idx]

    # Кривую мощности копируем, чтобы не менять исходные данные каталога.
    power_curve = base.power_curve.copy() if base.power_curve is not None else None
    if power_curve is not None and "power" in power_curve:
        power_curve = power_curve.copy()
        power_curve["power"] = power_curve["power"] * n_wind

    aggregated = WindTurbine(
        name=f"{base.name} x{n_wind}",
        nominal_power=base.nominal_power * n_wind,
        power_curve=power_curve,
        height=base.height,
        cut_in_speed=base.cut_in_speed,
        rated_speed=base.rated_speed,
        cut_out_speed=base.cut_out_speed,
    )

    return [aggregated]


def _aggregate_battery(batteries: List[Battery], batt_idx: int, n_batt: int) -> List[Battery]:
    """
    "Агрегирует" n_batt одинаковых АКБ в один эквивалентный объект Battery.

    Масштабируем:
    • capacity (кВт*ч) — линейно;
    • max_charge_power, max_discharge_power (кВт) — линейно;
    • efficiency, soc_min, soc_max — оставляем как у базовой батареи.

    current_soc выставляется в 50% от soc_max (как стартовая точка).
    Это важно, потому что greedy использует BatteryState.from_battery(...),
    который стартует от current_soc/initial_soc_kwh.
    """
    if not batteries or n_batt <= 0:
        return []

    base = batteries[batt_idx]
    aggregated = Battery(
        name=f"{base.name} x{n_batt}",
        capacity=base.capacity * n_batt,
        max_charge_power=base.max_charge_power * n_batt,
        max_discharge_power=base.max_discharge_power * n_batt,
        efficiency=base.efficiency,
        soc_min=base.soc_min,
        soc_max=base.soc_max,
    )
    aggregated.current_soc = aggregated.soc_max * 0.5

    return [aggregated]


def _ensure_diesel_columns(schedule: pd.DataFrame, diesel_unit_names: List[str]) -> pd.DataFrame:
    """
    Приводит schedule к единому виду по дизельной части, чтобы fitness считался стабильно.

    Проблема:
    • greedy может писать дизель как одной колонкой "diesel", или как поагрегатные колонки.
    • другие алгоритмы могут ожидать "diesel_total".
    • если колонок нет, fitness разваливается.

    Решение:
    • если поагрегатных колонок нет — добавляем нулевые колонки для каждого имени;
    • формируем diesel_series:
        - если есть "diesel" — берем ее;
        - иначе суммируем по поагрегатным колонкам;
        - иначе нулевой ряд.
    • гарантируем наличие колонок "diesel" и "diesel_total".
    """
    schedule = schedule.copy()
    diesel_columns = [col for col in schedule.columns if col in diesel_unit_names]

    if not diesel_columns and diesel_unit_names:
        for name in diesel_unit_names:
            if name not in schedule:
                schedule[name] = 0.0
        diesel_columns = diesel_unit_names

    diesel_series = None
    if "diesel" in schedule:
        diesel_series = schedule["diesel"]
    elif diesel_columns:
        diesel_series = schedule[diesel_columns].sum(axis=1)

    if diesel_series is None:
        diesel_series = pd.Series(np.zeros(len(schedule)), index=schedule.index)

    if "diesel" not in schedule:
        schedule["diesel"] = diesel_series

    if "diesel_total" not in schedule:
        schedule["diesel_total"] = diesel_series
    else:
        # Хак: "потрогать" колонку, чтобы совпадали индексы/типы.
        schedule["diesel_total"] = schedule["diesel_total"] + 0 * diesel_series

    # На всякий случай: если почему-то нет diesel, но есть diesel_total — копируем.
    if "diesel" not in schedule.columns and "diesel_total" in schedule.columns:
        schedule["diesel"] = schedule["diesel_total"]

    return schedule


def _evaluate_candidate(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_input: Optional[Sequence],
    position: np.ndarray,
    cache: Dict[Tuple[int, int, int, int, str], Tuple[float, OptimizationResult]],
    n_wind_max: int,
    n_batt_max: int,
    horizon: str,
) -> Tuple[float, OptimizationResult, Tuple[int, int, int, int]]:
    """
    Оценка одного кандидата (позиции волка):
    1) декодируем position -> (тип/кол-во ВЭУ, тип/кол-во АКБ);
    2) агрегируем оборудование в списки (по одному объекту ВЭУ/АКБ);
    3) запускаем greedy_optimization(...), получаем schedule;
    4) считаем fitness как штрафную функцию;
    5) кешируем результат по дискретному ключу.

    cache нужен, потому что разные волки/итерации могут попадать в одинаковые дискретные решения
    после округления position.
    """
    wind_idx, n_wind, batt_idx, n_batt = _decode_candidate(position, wind_turbines, batteries, n_wind_max, n_batt_max)
    key = (wind_idx, n_wind, batt_idx, n_batt, horizon)

    if key in cache:
        fitness, cached_result = cache[key]
        return fitness, cached_result, (wind_idx, n_wind, batt_idx, n_batt)

    # Собираем эквивалентные списки оборудования для dispatch.
    selected_wind = _aggregate_wind(wind_turbines, wind_idx, n_wind)
    selected_battery = _aggregate_battery(batteries, batt_idx, n_batt)

    # Запускаем нижний уровень (dispatch) -> получаем расписание.
    result = greedy_optimization(
        load_profile,
        diesel_plants,
        hydro_plants,
        selected_wind,
        selected_battery,
        wind_input,
        verbose=False,
    )

    # Приводим schedule к единому виду по дизелю.
    schedule = _ensure_diesel_columns(result.schedule, _get_diesel_unit_names(diesel_plants))

    # Считаем дизельный ряд.
    diesel_columns = [col for col in schedule.columns if col in _get_diesel_unit_names(diesel_plants)]
    diesel_series = schedule.get("diesel")
    if diesel_series is None and diesel_columns:
        diesel_series = schedule[diesel_columns].sum(axis=1)
    elif diesel_series is None:
        diesel_series = pd.Series(np.zeros(len(schedule)), index=schedule.index)

    # Прокси "энергии дизеля" как сумма мощности по часам (dt=1).
    diesel_total_energy = float(diesel_series.sum())

    # Сколько часов дизель "включен".
    diesel_hours = int((diesel_series > 0).sum()) if not diesel_series.empty else 0

    # Недоотпуск.
    unserved_series = schedule.get("unserved")
    if unserved_series is None:
        unserved_series = schedule.get("unserved_load_kw")

    # Шаг dt (если в schedule есть dt, учитываем его).
    dt_series = schedule.get("dt", 1.0)
    if np.isscalar(dt_series):
        dt_value = float(dt_series)
        dt_series = pd.Series(dt_value, index=schedule.index)

    if unserved_series is not None:
        unserved_energy = float((unserved_series * dt_series).sum())
    else:
        # Если нет колонки unserved — считаем это катастрофой (бесконечный штраф).
        unserved_energy = float("inf")

    # Расход топлива: берем то, что посчитал greedy (если есть), иначе fallback на diesel_total_energy.
    fuel_total = float(getattr(result, "total_fuel_consumption", diesel_total_energy))

    # Curtailment/dump: ищем первую подходящую колонку.
    curtailment_series = None
    for col in ("dump", "ballast", "curtailment"):
        if col in schedule:
            curtailment_series = schedule[col]
            break
    curtailment = float(curtailment_series.sum()) if curtailment_series is not None else 0.0

    # dump_energy (как сумма по колонке dump; при dt=1 это эквивалентно энергии).
    dump_energy = float(schedule.get("dump", pd.Series(np.zeros(len(schedule)), index=schedule.index)).sum())

    # CAPEX proxy: штраф за "размер" состава.
    capex_proxy = 0.0
    if wind_turbines and n_wind > 0:
        capex_proxy += WIND_CAPEX_WEIGHT * (n_wind * wind_turbines[wind_idx].nominal_power)
    if batteries and n_batt > 0:
        capex_proxy += BATTERY_CAPEX_WEIGHT * (n_batt * batteries[batt_idx].capacity)

    # Итоговая штрафная функция (ориентация: минимизация).
    fitness = (
        1e9 * unserved_energy
        + fuel_total
        + 5.0 * dump_energy
        + 1e3 * diesel_hours
        + capex_proxy
    )

    if not np.isfinite(fitness):
        fitness = float("inf")

    cache[key] = (fitness, result)
    return fitness, result, (wind_idx, n_wind, batt_idx, n_batt)


def _find_worst_window(load_profile: LoadProfile, window: int = 168) -> Tuple[int, LoadProfile]:
    """
    Находит "худшее" окно длиной window часов по критерию максимальной СУММЫ нагрузки.

    Возвращает:
    • start_idx — индекс начала окна;
    • LoadProfile(data=window_df) — укороченный профиль нагрузки.

    Это используется для FAST оценки: "самая тяжелая неделя" по суммарной энергии нагрузки.
    """
    load_series = _get_load_series(load_profile)
    if load_series.empty or len(load_series) <= window:
        return 0, load_profile

    rolling_sum = load_series.rolling(window, min_periods=window).sum()
    start_idx = int(np.argmax(rolling_sum.values) - window + 1)
    start_idx = max(0, min(start_idx, len(load_series) - window))
    end_idx = start_idx + window

    window_df = load_profile.data.iloc[start_idx:end_idx].copy()
    return start_idx, LoadProfile(data=window_df)


def _slice_wind_input(wind_input: Optional[Sequence], start: int, end: int) -> Optional[Sequence]:
    """
    Режет вход по ветру (список скоростей или DataFrame) на тот же временной интервал,
    что и выбранное FAST окно нагрузки.

    Нужно, чтобы greedy на FAST окне использовал согласованные данные нагрузки и ветра.
    """
    if wind_input is None:
        return None
    try:
        if isinstance(wind_input, pd.DataFrame):
            return wind_input.iloc[start:end]
        return wind_input[start:end]
    except Exception:
        # Если что-то пошло не так — возвращаем как есть (лучше, чем падать).
        return wind_input


def wolf_optimization(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_speeds: List[float] = None,
    wind_data: pd.DataFrame = None,
    n_wolves: int = 20,
    max_iter: int = 40,
    seed: int = 42,
    **kwargs,
) -> OptimizationResult:
    """
    Оптимизация методом серых волков (GWO).

    Вход:
    • load_profile, diesel_plants, hydro_plants — среда задачи (нагрузка и неизменяемые источники);
    • wind_turbines, batteries — каталоги ВЭУ и АКБ (из них выбираем типы/количества);
    • wind_speeds или wind_data — входной ветер (один из них);
    • n_wolves — размер популяции (количество волков);
    • max_iter — число итераций обновления позиций;
    • seed — фиксируем генератор случайных чисел для воспроизводимости;
    • kwargs:
        - progress_cb: колбэк прогресса.

    Выход:
    • лучший найденный OptimizationResult (желательно после FULL проверки).
    """

    progress_cb = kwargs.get("progress_cb")

    # wind_input_full — единый "источник данных ветра":
    # • если передали последовательность скоростей — используем ее;
    # • иначе используем DataFrame wind_data.
    wind_input_full = wind_speeds if wind_speeds is not None else wind_data

    # Верхние пределы количества оборудования.
    n_wind_max, n_batt_max = _compute_limits(load_profile, wind_turbines, batteries)

    # dim — размерность вектора решения:
    # • для ВЭУ: (индекс типа, количество) -> +2
    # • для АКБ: (индекс типа, количество) -> +2
    dim = 0
    if wind_turbines:
        dim += 2
    if batteries:
        dim += 2

    # Если нечего оптимизировать (нет ВЭУ и АКБ) — просто запускаем greedy dispatch как есть.
    if dim == 0:
        return greedy_optimization(
            load_profile,
            diesel_plants,
            hydro_plants,
            wind_turbines,
            batteries,
            wind_input_full,
            verbose=False,
        )

    # Границы поискового пространства.
    lower_bounds = np.zeros(dim)

    # upper_bounds хранится как список, потом превращается в np.array.
    # Для типовых индексов верхняя граница = len(catalog)-1, для количеств = n_*_max.
    upper_bounds: List[float] = []
    if wind_turbines:
        upper_bounds.extend([len(wind_turbines) - 1, max(n_wind_max, 0)])
    if batteries:
        upper_bounds.extend([len(batteries) - 1, max(n_batt_max, 0)])
    upper_bounds_arr = np.array(upper_bounds, dtype=float)

    # Инициализируем позиции волков равномерно в пределах границ.
    rng = np.random.default_rng(seed)
    positions = rng.uniform(lower_bounds, upper_bounds_arr, size=(n_wolves, dim))

    # ------------------------------ FAST окно (худшая неделя) ------------------------------
    fast_start, load_profile_fast = _find_worst_window(load_profile)
    fast_end = fast_start + len(load_profile_fast.data)
    wind_input_fast = _slice_wind_input(wind_input_full, fast_start, fast_end)

    # cache: ключ (wind_idx, n_wind, batt_idx, n_batt, horizon) -> (fitness, result)
    cache: Dict[Tuple[int, int, int, int, str], Tuple[float, OptimizationResult]] = {}

    # Лучшие решения на FULL и FAST горизонтах.
    best_full_result: Optional[OptimizationResult] = None
    best_full_fitness = float("inf")
    best_fast_result: Optional[OptimizationResult] = None
    best_fast_fitness = float("inf")

    # Параметры стратегии FAST/FULL.
    full_interval = 5  # раз в сколько итераций выполнять блок FULL проверок
    top_k = 15         # сколько лучших уникальных кандидатов (по дискретному decoded) проверять на FULL
    patience = 4       # сколько "полных" проверок без улучшения терпим, прежде чем остановиться
    no_improve_full = 0

    # Для прогресса: сколько всего fast оценок планируется.
    total_fast_evals = max_iter * n_wolves
    completed_fast = 0

    for iter_idx in range(max_iter):
        # Сообщаем прогресс итераций (грубая шкала).
        if progress_cb:
            progress_cb(iter_idx, max_iter, f"GWO fast iter {iter_idx + 1}/{max_iter}")

        # Параметр 'a' в GWO уменьшается линейно от 2 к 0:
        # • в начале поиск более "разбросанный";
        # • к концу — более "эксплуатация" вокруг лидеров.
        a = 2 - 2 * iter_idx / (max_iter - 1) if max_iter > 1 else 2

        fitness_values = []  # будем хранить (fitness, pos_copy, decoded) по всем волкам на текущей итерации

        # ------------------------------ Оценка каждого волка на FAST окне ------------------------------
        for pos in positions:
            fitness, result, decoded = _evaluate_candidate(
                load_profile_fast,
                diesel_plants,
                hydro_plants,
                wind_turbines,
                batteries,
                wind_input_fast,
                pos,
                cache,
                n_wind_max,
                n_batt_max,
                horizon="fast",
            )
            fitness_values.append((fitness, pos.copy(), decoded))

            # Обновляем лучший FAST результат.
            if fitness < best_fast_fitness:
                best_fast_fitness = fitness
                best_fast_result = result

            completed_fast += 1
            if progress_cb and (completed_fast % 10 == 0 or completed_fast == total_fast_evals):
                progress_cb(completed_fast, total_fast_evals, f"GWO fast iter {iter_idx + 1}/{max_iter}")

        # ------------------------------ Тройка лидеров: alpha, beta, delta ------------------------------
        # Чем меньше fitness, тем лучше (минимизация).
        fitness_array = np.array([f[0] for f in fitness_values])
        sorted_indices = np.argsort(fitness_array)

        # alpha — лучший, beta — второй, delta — третий.
        alpha_pos = positions[sorted_indices[0]].copy()
        beta_pos = positions[sorted_indices[1]].copy() if len(sorted_indices) > 1 else alpha_pos.copy()
        delta_pos = positions[sorted_indices[2]].copy() if len(sorted_indices) > 2 else alpha_pos.copy()

        # ------------------------------ Обновление позиций по формулам GWO ------------------------------
        for i in range(n_wolves):
            # Для каждого "лидера" (alpha/beta/delta) генерируются свои r1,r2,
            # из них считаются A и C, затем строятся три "предложения" X1,X2,X3,
            # и новая позиция — среднее от них.

            # --- Влияние alpha ---
            r1_alpha, r2_alpha = rng.random(dim), rng.random(dim)
            A1 = 2 * a * r1_alpha - a
            C1 = 2 * r2_alpha
            D_alpha = np.abs(C1 * alpha_pos - positions[i])
            X1 = alpha_pos - A1 * D_alpha

            # --- Влияние beta ---
            r1_beta, r2_beta = rng.random(dim), rng.random(dim)
            A2 = 2 * a * r1_beta - a
            C2 = 2 * r2_beta
            D_beta = np.abs(C2 * beta_pos - positions[i])
            X2 = beta_pos - A2 * D_beta

            # --- Влияние delta ---
            r1_delta, r2_delta = rng.random(dim), rng.random(dim)
            A3 = 2 * a * r1_delta - a
            C3 = 2 * r2_delta
            D_delta = np.abs(C3 * delta_pos - positions[i])
            X3 = delta_pos - A3 * D_delta

            # Итоговая позиция — среднее трех "векторов притяжения".
            new_position = (X1 + X2 + X3) / 3.0

            # Ограничиваем позицию границами (чтобы индексы/кол-ва не выходили за допустимое).
            new_position = np.minimum(np.maximum(new_position, lower_bounds), upper_bounds_arr)
            positions[i] = new_position

        # ------------------------------ FULL проверки (дорого) ------------------------------
        # Раз в full_interval итераций, а также на последней итерации, прогоняем
        # top_k лучших уникальных дискретных кандидатов на полном горизонте.
        if (iter_idx + 1) % full_interval == 0 or iter_idx == max_iter - 1:
            # Сначала собираем top_k уникальных decoded-кандидатов из текущего fitness_values.
            unique_candidates: Dict[Tuple[int, int, int, int], Tuple[float, np.ndarray]] = {}
            for fitness, pos, decoded in sorted(fitness_values, key=lambda x: x[0]):
                key = tuple(decoded)
                if key not in unique_candidates:
                    unique_candidates[key] = (fitness, pos)
                if len(unique_candidates) >= top_k:
                    break

            full_candidates = list(unique_candidates.values())

            # Прогоняем каждого на full горизонте.
            for idx_full, (fast_fit, pos) in enumerate(full_candidates):
                fitness_full, result_full, _ = _evaluate_candidate(
                    load_profile,
                    diesel_plants,
                    hydro_plants,
                    wind_turbines,
                    batteries,
                    wind_input_full,
                    pos,
                    cache,
                    n_wind_max,
                    n_batt_max,
                    horizon="full",
                )
                if progress_cb:
                    progress_cb(idx_full + 1, len(full_candidates), f"GWO full eval {idx_full + 1}/{len(full_candidates)}")

                # Обновляем лучший FULL результат и счетчик "без улучшений".
                if fitness_full < best_full_fitness:
                    best_full_fitness = fitness_full
                    best_full_result = result_full
                    no_improve_full = 0
                else:
                    no_improve_full += 1

            # Ранняя остановка: если несколько FULL проверок подряд не дают улучшения — выходим.
            if no_improve_full >= patience:
                break

    # ------------------------------ Возврат результата ------------------------------
    # Хотим вернуть лучший FULL (самый надежный).
    # Если FULL так и не нашли — возвращаем лучший FAST, иначе fallback на greedy без подбора.
    if best_full_result is None:
        if best_fast_result is not None:
            best_full_result = best_fast_result
        else:
            best_full_result = greedy_optimization(
                load_profile,
                diesel_plants,
                hydro_plants,
                wind_turbines,
                batteries,
                wind_input_full,
                verbose=False,
            )

    return best_full_result
