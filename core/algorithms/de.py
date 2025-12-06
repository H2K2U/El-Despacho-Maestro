# File: core/algorithms/de.py
"""
Оптимизация подбора состава оборудования методом дифференциальной эволюции (DE).

1) Что делает этот модуль
   Этот файл реализует алгоритм верхнего уровня "подбор состава" (sizing):
   • выбираем тип и количество ВЭУ (ветроустановок);
   • выбираем тип и количество АКБ (накопителей);
   • для каждого кандидата запускаем нижний уровень (диспетчеризацию) greedy_optimization;
   • по результату диспетчеризации считаем целевую (fitness) и минимизируем ее DE.

2) Математическая модель (в терминах "целевая функция / ограничения")

   2.1) Переменные верхнего уровня (дискретные)
   • wind_idx ∈ {0, 1, ..., N_w - 1} — индекс типа ВЭУ из каталога wind_turbines;
   • n_wind   ∈ {0, 1, ..., n_wind_max} — количество ВЭУ заданного типа;
   • batt_idx ∈ {0, 1, ..., N_b - 1} — индекс типа АКБ из каталога batteries;
   • n_batt   ∈ {0, 1, ..., n_batt_max} — количество АКБ заданного типа.

   В DE эти дискретные переменные кодируются вещественным вектором position и затем приводятся
   к целым значениям через round + ограничение диапазоном (clamp).

   2.2) Нижний уровень (диспетчеризация, почасовая) и его "внутренние" переменные
   greedy_optimization по заданному составу строит расписание schedule, где по часам (t):
   • P_diesel(t), P_hydro(t), P_wind(t) — выдача источников;
   • P_charge(t), P_discharge(t) — заряд и разряд АКБ;
   • SOC(t) — состояние заряда АКБ (в твоем проекте часто в кВт*ч);
   • P_unserved(t) ≥ 0 — недоотпуск нагрузки;
   • P_dump(t) ≥ 0 — сброс/балласт/curtailment (если образуется избыток).

   2.3) Целевая функция (минимизация)
   Здесь используется штрафная целевая "чем меньше, тем лучше":

   J =
      1e9 * E_unserved
    + Fuel_total
    + W_DUMP   * E_dump
    + W_HOURS  * H_diesel
    + W_STARTS * S_diesel
    + W_OVERLAP* O_overlap
    + CapexProxy

   где:
   • E_unserved = Σ_t (unserved(t) * dt(t)) — энергия недоотпуска (очень большой штраф 1e9);
   • Fuel_total — расход топлива (из result.total_fuel_consumption, либо прокси);
   • E_dump     = Σ_t dump(t) (+ ballast/curtailment при наличии);
   • H_diesel   = количество часов (шагов), где diesel(t) > 0;
   • S_diesel   = количество стартов дизеля (переход 0 -> >0);
   • O_overlap  = Σ_t min(charge_power(t), discharge_power(t)) — "перекрытие" заряд/разряд;
   • CapexProxy — грубая прокси капитальных затрат (может быть отключена весами).

   2.4) Ограничения
   Явные (верхний уровень):
   • индексы типов ограничены размерами каталогов;
   • количества ограничены n_wind_max и n_batt_max, которые выводятся из пика нагрузки и минимальных
     мощностей оборудования + жесткий верхний предел 200.
   Неявные (нижний уровень, реализованы greedy_optimization):
   • баланс мощности по часу;
   • ограничения по мощности источников и АКБ;
   • динамика SOC и границы SOC_min..SOC_max;
   • учет КПД АКБ;
   • недоотпуск и сброс не отрицательны.

3) FAST/FULL стратегия ускорения
   Чтобы DE не считал на всем горизонте всегда:
   • FAST: оцениваем кандидата на нескольких "плохих" окнах по 168 часов (7 суток);
   • FULL: периодически делаем дорогую проверку лучших кандидатов на полном горизонте.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Нижний уровень (dispatch): из состава оборудования строит расписание (schedule) и метрики результата.
from core.algorithms.greedy import greedy_optimization

# Модели домена (оборудование, профиль нагрузки, результат оптимизации).
from core.models import Battery, DieselPlant, HydroPlant, LoadProfile, OptimizationResult, WindTurbine

# ------------------------------ Весовые коэффициенты целевой функции (штрафы) ------------------------------
# Все коэффициенты ниже используются в _fitness_from_result.
# Важно: fitness здесь "чем меньше, тем лучше" (задача минимизации).

# Штраф за "сброс" (dump/ballast/curtailment).
W_DUMP = 10.0

# Штраф за количество часов работы дизеля (diesel > 0).
W_HOURS = 1e3

# Штраф за количество стартов дизеля (переход 0 -> >0).
W_STARTS = 1e4

# Очень большой штраф за одновременный заряд и разряд АКБ (признак физически плохого режима/ошибки).
W_OVERLAP = 1e6

# Прокси CAPEX (пока 0.0 = отключено).
W_WIND_CAPEX = 0.0
W_BATT_CAPEX = 0.0

# Если в каталогах нет данных или данные нулевые, используем безопасное значение мощности "по умолчанию".
DEFAULT_MIN_POWER = 50.0

# Длина FAST-окна: 168 часов = 7 суток.
FAST_WINDOW_HOURS = 168


def _clamp(value: float, lower: float, upper: float) -> float:
    """
    Ограничение значения диапазоном [lower; upper].

    value — то, что нужно ограничить;
    lower — нижняя граница;
    upper — верхняя граница.

    Возвращает:
    • lower, если value < lower;
    • upper, если value > upper;
    • value, если lower <= value <= upper.
    """
    return max(lower, min(upper, value))


def _get_load_series(load_profile: LoadProfile) -> pd.Series:
    """
    Пытается извлечь временной ряд нагрузки из LoadProfile в виде pandas.Series.

    Логика:
    • если load_profile.data — DataFrame и есть колонка "load", берем ее;
    • иначе берем первую числовую колонку как запасной вариант;
    • если ничего не найдено — возвращаем пустой Series(dtype=float).

    Зачем это нужно:
    • для вычисления пика нагрузки (ограничения по количеству оборудования);
    • для выбора FAST-окон (по сумме нагрузки и по пику).
    """
    if isinstance(load_profile.data, pd.DataFrame):
        if "load" in load_profile.data.columns:
            return load_profile.data["load"]

        # Запасной путь: если "load" нет, пытаемся взять первую числовую колонку.
        numeric_cols = load_profile.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return load_profile.data[numeric_cols[0]]

    return pd.Series(dtype=float)


def _compute_limits(load_profile: LoadProfile, wind_turbines: List[WindTurbine], batteries: List[Battery]) -> Tuple[int, int]:
    """
    Оценивает верхние границы количества ВЭУ и АКБ, чтобы сузить пространство поиска.

    Шаги:
    1) p_peak = max(load) — пик нагрузки по профилю (кВт);
    2) min_wind_power = минимальная nominal_power среди каталога ВЭУ;
    3) min_batt_power = минимальная мощность АКБ (берем разряд, иначе заряд);
    4) n_wind_max ~= ceil(1.5 * p_peak / min_wind_power)
       n_batt_max ~= ceil(1.0 * p_peak / min_batt_power)
    5) дополнительно ограничиваем каждую границу сверху числом 200.

    Почему множители 1.5 и 1.0:
    • 1.5 для ВЭУ дает запас по установленной мощности (иначе DE может "не дотянуться" до разумных решений);
    • 1.0 для АКБ ограничивает ее по мощности примерно на уровне пика.

    Возвращает:
    (n_wind_max, n_batt_max).
    """
    load_series = _get_load_series(load_profile)
    p_peak = float(load_series.max()) if not load_series.empty else 0.0

    # Минимальная мощность одной ВЭУ из каталога (или DEFAULT_MIN_POWER, если каталог пуст).
    min_wind_power = min((wt.nominal_power for wt in wind_turbines), default=DEFAULT_MIN_POWER)

    # Минимальная мощность АКБ:
    # • приоритет max_discharge_power (важно для покрытия нагрузки);
    # • если его нет, берем max_charge_power как запасной параметр;
    # • если и его нет, получаем 0 и потом заменяем на DEFAULT_MIN_POWER.
    min_batt_power = min(
        (getattr(bt, "max_discharge_power", None) or getattr(bt, "max_charge_power", 0) or 0) for bt in batteries
    ) if batteries else DEFAULT_MIN_POWER
    if min_batt_power == 0:
        min_batt_power = DEFAULT_MIN_POWER

    # Приближенные верхние границы по количеству оборудования.
    n_wind_max = int(np.ceil(1.5 * p_peak / min_wind_power)) if min_wind_power > 0 else 0
    n_batt_max = int(np.ceil(1.0 * p_peak / min_batt_power)) if min_batt_power > 0 else 0

    # Ограничение диапазоном [0; 200], чтобы поиск не раздувался до нереальных значений.
    return min(max(n_wind_max, 0), 200), min(max(n_batt_max, 0), 200)


def _decode_candidate(
    position: np.ndarray,
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    n_wind_max: int,
    n_batt_max: int,
) -> Tuple[int, int, int, int]:
    """
    Декодирует вещественный вектор position (генерируемый DE) в дискретный кандидат состава:

    Возвращает:
    (wind_idx, n_wind, batt_idx, n_batt)

    Структура position зависит от доступных каталогов:
    • если wind_turbines не пуст: используем 2 координаты: wind_idx и n_wind;
    • если batteries не пуст: используем 2 координаты: batt_idx и n_batt;
    Итого dim = 0 / 2 / 4.

    Как превращаем вещественное -> целое:
    • clamp в допустимый диапазон;
    • round до ближайшего целого;
    • cast в int.
    """
    idx = 0  # текущая позиция чтения из position
    wind_idx = 0
    n_wind = 0
    batt_idx = 0
    n_batt = 0

    # Декодирование части, относящейся к ВЭУ: [wind_idx, n_wind]
    if wind_turbines:
        wind_idx = int(round(_clamp(position[idx], 0, len(wind_turbines) - 1)))
        idx += 1
        n_wind = int(round(_clamp(position[idx], 0, n_wind_max)))
        idx += 1

    # Декодирование части, относящейся к АКБ: [batt_idx, n_batt]
    if batteries:
        batt_idx = int(round(_clamp(position[idx], 0, len(batteries) - 1)))
        idx += 1
        n_batt = int(round(_clamp(position[idx], 0, n_batt_max)))

    return wind_idx, n_wind, batt_idx, n_batt


def _aggregate_wind(wind_turbines: List[WindTurbine], wind_idx: int, n_wind: int) -> List[WindTurbine]:
    """
    Агрегирует "тип + количество" ВЭУ в список WindTurbine для диспетчеризации.

    Идея:
    • вместо n одинаковых объектов создаем один агрегированный объект;
    • номинальная мощность умножается на n;
    • кривая мощности (power_curve), если существует и содержит колонку "power",
      также масштабируется по мощности в n раз.

    Возвращает:
    • [] если ВЭУ нет или n_wind <= 0;
    • иначе список из одного агрегированного WindTurbine.
    """
    if not wind_turbines or n_wind <= 0:
        return []

    base = wind_turbines[wind_idx]  # выбранный тип ВЭУ
    power_curve = base.power_curve.copy() if base.power_curve is not None else None

    # Масштабируем кривую мощности (если она есть и внутри есть колонка "power").
    if power_curve is not None and "power" in power_curve:
        power_curve = power_curve.copy()
        power_curve["power"] = power_curve["power"] * n_wind

    aggregated = WindTurbine(
        # Имя делаем человекочитаемым: "название x количество"
        name=f"{base.name} x{n_wind}",

        # Суммарная номинальная мощность (кВт)
        nominal_power=base.nominal_power * n_wind,

        # Суммарная кривая мощности
        power_curve=power_curve,

        # Остальные параметры оставляем как у базовой ВЭУ
        height=base.height,
        cut_in_speed=base.cut_in_speed,
        rated_speed=base.rated_speed,
        cut_out_speed=base.cut_out_speed,
    )
    return [aggregated]


def _aggregate_battery(batteries: List[Battery], batt_idx: int, n_batt: int) -> List[Battery]:
    """
    Агрегирует "тип + количество" АКБ в список Battery для диспетчеризации.

    Идея:
    • вместо n одинаковых объектов создаем один агрегированный объект;
    • емкость и предельные мощности заряда/разряда умножаются на n;
    • КПД и границы SOC (soc_min/soc_max) берутся как у базовой АКБ.

    Важно про current_soc:
    • мы вручную задаем current_soc = 0.5 * soc_max (условно "50% от верхней границы"),
      чтобы кандидат начинал оценку из фиксированной точки.
      Это влияет на сравнимость кандидатов.
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

    # Начальный SOC задаем в середине допустимого диапазона (по верхней границе).
    aggregated.current_soc = aggregated.soc_max * 0.5
    return [aggregated]


def _get_diesel_unit_names(diesel_plants: List[DieselPlant]) -> List[str]:
    """
    Формирует список ожидаемых имен дизельных агрегатов (ДГУ) для дальнейшей нормализации schedule.

    Зачем:
    • иногда schedule содержит отдельные колонки по каждому ДГУ;
    • иногда только суммарную колонку "diesel" или "diesel_total";
    • чтобы унифицировать вычисления часов, стартов и т.д., мы пытаемся восстановить ожидаемые имена.

    Логика:
    • если unit.name пустое, присваиваем "ДГУ 1", "ДГУ 2", ...;
    • если plant.name задано, делаем "PlantName - UnitName".
    """
    names: List[str] = []
    unit_counter = 1  # счетчик для автогенерации имени ДГУ
    for plant in diesel_plants:
        for unit in plant.diesel_units:
            unit_name = unit.name or f"ДГУ {unit_counter}"
            full_name = f"{plant.name} - {unit_name}" if plant.name else unit_name
            names.append(full_name)
            unit_counter += 1
    return names


def _ensure_diesel_columns(schedule: pd.DataFrame, diesel_unit_names: List[str]) -> pd.DataFrame:
    """
    Приводит DataFrame schedule к унифицированному виду по дизельным колонкам.

    Делает:
    1) находим колонки, которые совпадают с ожидаемыми именами ДГУ;
    2) если ни одной нет, но список имен ДГУ известен, создаем эти колонки и заполняем нулями;
    3) вычисляем diesel_series:
       • если есть "diesel", берем его;
       • иначе суммируем по дизельным колонкам (поагрегатным);
       • иначе создаем нулевой ряд;
    4) гарантируем наличие колонок "diesel" и "diesel_total".

    Почему есть и "diesel", и "diesel_total":
    • разная логика в разных частях проекта/версии результата;
    • наличие обеих колонок снижает шанс KeyError и упрощает последующую обработку.
    """
    schedule = schedule.copy()
    diesel_columns = [col for col in schedule.columns if col in diesel_unit_names]

    # Если поагрегатных колонок нет, но мы знаем, как они могли называться, создаем их нулями.
    if not diesel_columns and diesel_unit_names:
        for name in diesel_unit_names:
            if name not in schedule:
                schedule[name] = 0.0
        diesel_columns = diesel_unit_names

    # Определяем суммарный ряд дизеля.
    diesel_series = None
    if "diesel" in schedule:
        diesel_series = schedule["diesel"]
    elif diesel_columns:
        diesel_series = schedule[diesel_columns].sum(axis=1)

    # Если так и не нашли дизель, считаем, что он нулевой.
    if diesel_series is None:
        diesel_series = pd.Series(np.zeros(len(schedule)), index=schedule.index)

    # Гарантируем наличие унифицированных колонок.
    if "diesel" not in schedule:
        schedule["diesel"] = diesel_series
    if "diesel_total" not in schedule:
        schedule["diesel_total"] = diesel_series
    else:
        # Этот трюк не меняет значения, но позволяет "привязать" тип/индексацию.
        schedule["diesel_total"] = schedule["diesel_total"] + 0 * diesel_series

    return schedule


def _fitness_from_result(
    result: OptimizationResult,
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    diesel_unit_names: List[str],
    decoded: Tuple[int, int, int, int],
) -> float:
    """
    Вычисляет fitness (штрафную целевую) по результату диспетчеризации.

    decoded = (wind_idx, n_wind, batt_idx, n_batt) нужен, чтобы:
    • при необходимости добавить прокси CAPEX относительно выбранного оборудования.

    Главная идея:
    • недоотпуск (unserved) должен быть почти "запрещен" => огромный множитель 1e9;
    • дальше внутри допустимых решений минимизируем топливо и вторичные штрафы.
    """
    wind_idx, n_wind, batt_idx, n_batt = decoded

    # schedule ожидается DataFrame; если result.schedule не DataFrame, подставляем пустой.
    schedule = result.schedule if isinstance(result.schedule, pd.DataFrame) else pd.DataFrame()

    # Нормализуем дизельные колонки (diesel/diesel_total).
    schedule = _ensure_diesel_columns(schedule, diesel_unit_names)

    # --------------- Извлекаем ряд дизеля (diesel_series) ---------------
    # Пытаемся по приоритету: diesel_total -> diesel -> эвристика по именам колонок.
    diesel_series = schedule.get("diesel_total")
    if diesel_series is None:
        diesel_series = schedule.get("diesel")
    if diesel_series is None:
        # Эвристика: если названия колонок содержат "дгу" или "diesel", суммируем их.
        diesel_cols = [col for col in schedule.columns if "дгу" in col.lower() or "diesel" in col.lower()]
        if diesel_cols:
            diesel_series = schedule[diesel_cols].sum(axis=1)
    if diesel_series is None:
        # Если вообще ничего нет, считаем дизель нулевым (иначе дальше посыпятся ошибки).
        diesel_series = pd.Series(np.zeros(len(schedule)), index=schedule.index)

    # --------------- Извлекаем недоотпуск (unserved_series) ---------------
    # Поддерживаем две возможные схемы именования: "unserved" и "unserved_load_kw".
    unserved_series = None
    for col in ("unserved", "unserved_load_kw"):
        if col in schedule:
            unserved_series = schedule[col]
            break

    # --------------- Извлекаем шаг времени dt ---------------
    # dt может быть скаляром (одно значение) или временным рядом.
    dt_series = schedule.get("dt", 1.0)
    if np.isscalar(dt_series):
        dt_value = float(dt_series)
        dt_series = pd.Series(dt_value, index=schedule.index)

    # Энергия недоотпуска: Σ(unserved * dt).
    # Если нет колонки unserved, ставим inf, чтобы кандидат считался недопустимым.
    if unserved_series is not None:
        unserved_energy = float((unserved_series * dt_series).sum())
    else:
        unserved_energy = float("inf")

    # --------------- Сброс (dump) и его синонимы ---------------
    # dump_energy берем как сумму "dump" (если нет, нули).
    dump_energy = float(schedule.get("dump", pd.Series(np.zeros(len(schedule)), index=schedule.index)).sum())

    # Иногда сброс называют "ballast" или "curtailment" (берем один из них, если присутствует).
    curtailment = None
    for col in ("ballast", "curtailment"):
        if col in schedule:
            curtailment = schedule[col]
            break
    if curtailment is not None:
        dump_energy += float(curtailment.sum())

    # --------------- Часы работы и старты дизеля ---------------
    # Часы: сколько шагов дизель выдавал положительную мощность.
    diesel_hours = int((diesel_series > 0).sum()) if not diesel_series.empty else 0

    # Старты: сколько раз дизель переходил из 0 в >0.
    diesel_starts = 0
    if len(diesel_series) > 1:
        diesel_starts = int(((diesel_series.shift(1, fill_value=0) == 0) & (diesel_series > 0)).sum())

    # --------------- Расход топлива ---------------
    # Если у OptimizationResult есть total_fuel_consumption, используем его.
    # Иначе fallback: сумма diesel_series (это прокси и не всегда единицы совпадают, но лучше чем ничего).
    fuel_total = float(getattr(result, "total_fuel_consumption", diesel_series.sum()))

    # --------------- Перекрытие заряд/разряд АКБ ---------------
    # overlap = Σ min(charge_power, discharge_power).
    # Обычно заряд и разряд одновременно недопустимы или должны быть очень малы (из-за численных эффектов).
    overlap = 0.0
    if "charge_power" in schedule and "discharge_power" in schedule:
        overlap = float(np.minimum(schedule["charge_power"], schedule["discharge_power"]).sum())

    # --------------- Прокси CAPEX ---------------
    # Это грубое приближение: стоимость пропорциональна установленной мощности ВЭУ и емкости АКБ.
    # Сейчас веса W_WIND_CAPEX/W_BATT_CAPEX равны 0.0, то есть влияние отключено.
    capex_proxy = 0.0
    if wind_turbines and n_wind > 0:
        capex_proxy += W_WIND_CAPEX * (n_wind * wind_turbines[wind_idx].nominal_power)
    if batteries and n_batt > 0:
        capex_proxy += W_BATT_CAPEX * (n_batt * batteries[batt_idx].capacity)

    # --------------- Итоговый fitness ---------------
    # Важно: это минимизируемая величина.
    fitness = (
        1e9 * unserved_energy
        + fuel_total
        + W_DUMP * dump_energy
        + W_HOURS * diesel_hours
        + W_STARTS * diesel_starts
        + W_OVERLAP * overlap
        + capex_proxy
    )

    # Если получился NaN/inf по какой-то причине, приводим к inf.
    if not np.isfinite(fitness):
        fitness = float("inf")
    return fitness


def _evaluate_candidate(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_input: Optional[Sequence],
    position: np.ndarray,
    cache: Dict[Tuple[int, int, int, int, str, str], Tuple[float, OptimizationResult]],
    n_wind_max: int,
    n_batt_max: int,
    horizon_tag: str,
    window_tag: str,
) -> Tuple[float, OptimizationResult, Tuple[int, int, int, int]]:
    """
    Оценивает одного кандидата DE на заданном горизонте/окне.

    Шаги:
    1) Декодируем position -> (wind_idx, n_wind, batt_idx, n_batt);
    2) Формируем ключ кэша: (decoded + horizon_tag + window_tag);
    3) Если уже считали, возвращаем из кэша (ускорение);
    4) Агрегируем выбранные ВЭУ/АКБ (получаем список компонентов для greedy);
    5) Запускаем greedy_optimization и получаем OptimizationResult;
    6) Считаем fitness по результату;
    7) Кладем в кэш и возвращаем.

    Важно про кэш:
    • ключ зависит от окна, потому что один и тот же кандидат может вести себя по-разному на разных
      кусках года (FAST окна).
    """
    decoded = _decode_candidate(position, wind_turbines, batteries, n_wind_max, n_batt_max)
    key = (*decoded, horizon_tag, window_tag)

    # Если кандидат на таком же окне/горизонте уже считали, берем результат без перерасчета greedy.
    if key in cache:
        fitness, cached_result = cache[key]
        return fitness, cached_result, decoded

    wind_idx, n_wind, batt_idx, n_batt = decoded

    # Превращаем (тип, количество) -> список компонентов для диспетчеризации.
    selected_wind = _aggregate_wind(wind_turbines, wind_idx, n_wind)
    selected_batt = _aggregate_battery(batteries, batt_idx, n_batt)

    # Запуск диспетчеризации (нижний уровень). wind_input передаем как "факт по ветру".
    result = greedy_optimization(
        load_profile,
        diesel_plants,
        hydro_plants,
        selected_wind,
        selected_batt,
        wind_input,
        verbose=False,  # важно: при тысячах вызовов нельзя шуметь в консоль
    )

    # Имена ДГУ нужны для нормализации дизельных колонок в schedule.
    diesel_names = _get_diesel_unit_names(diesel_plants)

    # Расчет fitness по результату.
    fitness = _fitness_from_result(result, wind_turbines, batteries, diesel_names, decoded)

    # Сохраняем в кэш, чтобы повторно не считать тот же дискретный кандидат.
    cache[key] = (fitness, result)
    return fitness, result, decoded


def _slice_profile(load_profile: LoadProfile, start: int, end: int) -> LoadProfile:
    """
    Возвращает "подпрофиль" нагрузки [start:end] для FAST окна.

    Если load_profile.data не DataFrame, то мы не умеем корректно срезать и возвращаем как есть.
    """
    if not isinstance(load_profile.data, pd.DataFrame):
        return load_profile
    sliced_df = load_profile.data.iloc[start:end].copy()
    return LoadProfile(data=sliced_df)


def _slice_wind_input(wind_input: Optional[Sequence], start: int, end: int) -> Optional[Sequence]:
    """
    Возвращает "подвход" по ветру [start:end], синхронизированный с окном нагрузки.

    wind_input может быть:
    • DataFrame (режем через iloc);
    • list/np.ndarray/Sequence (режем через срез);
    • что-то еще (тогда ловим исключение и возвращаем wind_input как есть).
    """
    if wind_input is None:
        return None
    try:
        if isinstance(wind_input, pd.DataFrame):
            return wind_input.iloc[start:end]
        return wind_input[start:end]
    except Exception:
        return wind_input


def _select_fast_windows(load_profile: LoadProfile) -> List[Tuple[int, int, str]]:
    """
    Выбирает FAST-окна длиной FAST_WINDOW_HOURS, которые считаются "тяжелыми" для системы.

    Возвращает список окон:
    • (start_sum,  start_sum + window,  "fast_sum")  — окно с максимальной суммой нагрузки (энергетически тяжелое);
    • (start_max,  start_max + window,  "fast_peak") — окно с максимальным пиком нагрузки (мощностной стресс).

    Если данных меньше window, возвращаем одно окно "full" на весь диапазон.
    """
    load_series = _get_load_series(load_profile)
    if load_series.empty:
        # Если нет данных, возвращаем полный диапазон как единственное "окно".
        return [(0, len(load_profile.data), "full")]

    window = FAST_WINDOW_HOURS
    n = len(load_series)
    if n <= window:
        return [(0, n, "full")]

    # Скользящая сумма по окну (min_periods=window означает: пока окно не заполнено, будет NaN).
    rolling_sum = load_series.rolling(window, min_periods=window).sum()
    # Находим индекс максимума и переводим его в старт окна.
    start_sum = int(np.argmax(rolling_sum.values) - window + 1)
    start_sum = max(0, min(start_sum, n - window))

    # Скользящий максимум по окну.
    rolling_max = load_series.rolling(window, min_periods=window).max()
    start_max = int(np.argmax(rolling_max.values) - window + 1)
    start_max = max(0, min(start_max, n - window))

    return [
        (start_sum, start_sum + window, "fast_sum"),
        (start_max, start_max + window, "fast_peak"),
    ]


def _evaluate_fast(
    load_profile: LoadProfile,
    windows: List[Tuple[int, int, str]],
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_input: Optional[Sequence],
    position: np.ndarray,
    cache: Dict[Tuple[int, int, int, int, str, str], Tuple[float, OptimizationResult]],
    n_wind_max: int,
    n_batt_max: int,
) -> Tuple[float, OptimizationResult, Tuple[int, int, int, int]]:
    """
    FAST-оценка кандидата: прогоняем кандидата на нескольких коротких окнах и возвращаем "самую плохую" оценку.

    Почему "самую плохую":
    • если кандидат проваливается на одном из стрессовых окон, то на полном горизонте он тоже рискован;
    • поэтому разумно принимать оценку FAST как max(fitness) по окнам (пессимистичная оценка).

    Важно:
    • фитнес у нас минимизируемый (меньше лучше), но здесь мы берем МАКСИМУМ по окнам.
      Это не ошибка само по себе: это ровно "worst-case" по выбранным окнам.
      Однако название переменной best_fitness может путать: фактически это "worst_fitness" по окнам.
    """
    best_fitness = -float("inf")  # стартуем снизу, чтобы найти максимум по окнам
    best_result: Optional[OptimizationResult] = None
    best_decoded: Tuple[int, int, int, int] = (0, 0, 0, 0)

    for start, end, tag in windows:
        # Срезаем профиль нагрузки и ветровые данные под конкретное FAST окно.
        sub_profile = _slice_profile(load_profile, start, end)
        sub_wind = _slice_wind_input(wind_input, start, end)

        # Оцениваем кандидата на этом окне.
        fitness, result, decoded = _evaluate_candidate(
            sub_profile,
            diesel_plants,
            hydro_plants,
            wind_turbines,
            batteries,
            sub_wind,
            position,
            cache,
            n_wind_max,
            n_batt_max,
            horizon_tag="fast",
            window_tag=tag,
        )

        # Выбираем наихудшее окно (максимальный fitness).
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = result
            best_decoded = decoded

    # Если по какой-то причине best_result не определился, возвращаем пустой результат (защита).
    return best_fitness, best_result if best_result is not None else OptimizationResult(schedule=pd.DataFrame()), best_decoded


def _evaluate_full(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_input: Optional[Sequence],
    position: np.ndarray,
    cache: Dict[Tuple[int, int, int, int, str, str], Tuple[float, OptimizationResult]],
    n_wind_max: int,
    n_batt_max: int,
) -> Tuple[float, OptimizationResult, Tuple[int, int, int, int]]:
    """
    FULL-оценка кандидата: запускаем диспетчеризацию на полном горизонте данных.

    В отличие от FAST:
    • не режем данные по окну;
    • window_tag фиксирован как "full";
    • horizon_tag = "full".
    """
    return _evaluate_candidate(
        load_profile,
        diesel_plants,
        hydro_plants,
        wind_turbines,
        batteries,
        wind_input,
        position,
        cache,
        n_wind_max,
        n_batt_max,
        horizon_tag="full",
        window_tag="full",
    )


def _progress(progress_cb, done: int, total: int, msg: str, last_report: int) -> int:
    """
    Удобная обертка для прогресс-колбэка, чтобы не дергать его слишком часто.

    progress_cb — функция вида progress_cb(done, total, msg) или None;
    done — сколько "оценок" (FAST+FULL) уже выполнено;
    total — общий условный бюджет (обычно max_fast_evals + max_full_evals);
    msg — текущий статус текстом;
    last_report — значение done, когда мы репортили прогресс в последний раз.

    Логика:
    • репортим каждые 5 выполненных оценок (или в самом конце).
    """
    if progress_cb is None:
        return last_report
    if done - last_report >= 5 or done == total:
        progress_cb(done, total, msg)
        return done
    return last_report


def _neighboring_indices(current_idx: int, count: int) -> Iterable[int]:
    """
    Генератор "соседних" индексов вокруг current_idx: +1, +2, -1, -2 в пределах [0; count-1].

    Используется в локальном поиске, чтобы попробовать близкие по каталогу типы оборудования.
    """
    if count <= 1:
        return []
    offsets = [1, 2, -1, -2]
    for offset in offsets:
        idx = current_idx + offset
        if 0 <= idx < count:
            yield idx


def _local_search(
    best_pos: np.ndarray,
    current_best_fitness: float,
    load_profile: LoadProfile,
    windows: List[Tuple[int, int, str]],
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_input: Optional[Sequence],
    cache: Dict[Tuple[int, int, int, int, str, str], Tuple[float, OptimizationResult]],
    n_wind_max: int,
    n_batt_max: int,
    rng: np.random.Generator,
    max_evals: int,
    eval_counter: List[int],
) -> Tuple[np.ndarray, float, Optional[OptimizationResult]]:
    """
    Небольшой локальный поиск около текущего лучшего кандидата на FAST.

    Идея:
    • DE хорошо исследует пространство, но иногда полезно "докрутить" решение локально;
    • здесь мы делаем несколько быстрых попыток вокруг best_pos:
      1) по каждой координате пробуем сдвиги -2, -1, +1, +2;
      2) дополнительно пробуем соседние индексы типов ВЭУ/АКБ в каталоге.

    Ограничения:
    • не более 12 проверок кандидатов за один вызов;
    • не превышаем общий лимит max_evals (контролируется через eval_counter[0]).
    """
    if eval_counter[0] >= max_evals:
        return best_pos, current_best_fitness, None

    candidates: List[np.ndarray] = []
    base = best_pos.copy()
    dim = len(base)

    # 1) "Пощупать" соседние значения всех координат.
    # Важно: координаты здесь еще вещественные; декодирование в целые делается позже через _decode_candidate.
    for idx in range(dim):
        for step in (-2, -1, 1, 2):
            cand = base.copy()
            cand[idx] = base[idx] + step
            candidates.append(cand)

    # 2) Дополнительно попробовать соседние ТИПЫ оборудования (по индексам каталога).
    offset = 0
    if wind_turbines:
        wind_idx = int(round(_clamp(base[offset], 0, len(wind_turbines) - 1)))
        for alt in _neighboring_indices(wind_idx, len(wind_turbines)):
            cand = base.copy()
            cand[offset] = alt
            candidates.append(cand)
        offset += 2  # пропускаем пару [wind_idx, n_wind]
    if batteries:
        batt_idx = int(round(_clamp(base[offset], 0, len(batteries) - 1))) if offset < len(base) else 0
        for alt in _neighboring_indices(batt_idx, len(batteries)):
            cand = base.copy()
            cand[offset] = alt
            candidates.append(cand)

    # Перемешиваем, чтобы порядок попыток не был детерминированным и не создавал перекос.
    rng.shuffle(candidates)

    tried = 0
    best_local_pos = base
    best_local_fit = current_best_fitness
    best_local_result: Optional[OptimizationResult] = None

    for cand in candidates:
        # Ограничение на число проб и общий бюджет оценок.
        if eval_counter[0] >= max_evals or tried >= 12:
            break

        fitness, res, _ = _evaluate_fast(
            load_profile,
            windows,
            diesel_plants,
            hydro_plants,
            wind_turbines,
            batteries,
            wind_input,
            cand,
            cache,
            n_wind_max,
            n_batt_max,
        )
        eval_counter[0] += 1
        tried += 1

        # Так как fitness минимизируем, улучшение это fitness < best_local_fit.
        if fitness < best_local_fit:
            best_local_fit = fitness
            best_local_pos = cand.copy()
            best_local_result = res

            # Поддерживаем актуальность current_best_fitness.
            if best_local_fit < current_best_fitness:
                current_best_fitness = best_local_fit

    return best_local_pos, best_local_fit, best_local_result


def de_optimization(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_speeds: Optional[Sequence] = None,
    wind_data: Optional[pd.DataFrame] = None,
    progress_cb=None,
    population_size: int = 25,
    generations: int = 40,
    mutation_factor: float = 0.7,
    crossover_rate: float = 0.9,
    seed: int = 42,
    max_fast_evals: int = 600,
    max_full_evals: int = 30,
    full_check_interval: int = 4,
    top_k: int = 15,
    **kwargs,
) -> OptimizationResult:
    """
    Дифференциальная эволюция (DE) для подбора состава оборудования с оценкой FAST/FULL.

    Ключевая идея:
    • DE генерирует кандидатов состава (тип/количество ВЭУ и АКБ);
    • каждый кандидат оценивается через запуск greedy_optimization (диспетчеризация);
    • цель: минимизировать штрафную функцию fitness (см. _fitness_from_result и модульный docstring).

    Параметры:
    • wind_speeds / wind_data — вход по ветру (берем wind_speeds, если он задан, иначе wind_data);
    • population_size — размер популяции DE;
    • generations — число поколений;
    • mutation_factor — базовый коэффициент мутации (F);
    • crossover_rate — вероятность кроссовера (CR);
    • max_fast_evals / max_full_evals — бюджеты вычислений для FAST и FULL;
    • full_check_interval — как часто делать FULL-проверку;
    • top_k — сколько уникальных decoded-кандидатов проверять в FULL.
    """
    # Выбираем вход по ветру: если есть wind_speeds, он приоритетнее, иначе берем wind_data.
    wind_input_full = wind_speeds if wind_speeds is not None else wind_data

    # Рассчитываем верхние границы количества ВЭУ и АКБ, чтобы ограничить пространство поиска.
    n_wind_max, n_batt_max = _compute_limits(load_profile, wind_turbines, batteries)

    # Размерность оптимизируемого вектора position.
    # По 2 координаты на каждую категорию:
    # • ВЭУ: [wind_idx, n_wind]
    # • АКБ: [batt_idx, n_batt]
    dim = 0
    if wind_turbines:
        dim += 2
    if batteries:
        dim += 2

    # Если оптимизировать нечего (нет ВЭУ и АКБ), просто запускаем greedy как есть.
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

    # Нижние границы: все координаты неотрицательные.
    lower_bounds = np.zeros(dim)

    # Верхние границы:
    # • для индексов типов: len(catalog) - 1;
    # • для количеств: n_wind_max / n_batt_max.
    upper_bounds: List[float] = []
    if wind_turbines:
        upper_bounds.extend([len(wind_turbines) - 1, max(n_wind_max, 0)])
    if batteries:
        upper_bounds.extend([len(batteries) - 1, max(n_batt_max, 0)])
    upper_bounds_arr = np.array(upper_bounds, dtype=float)

    # Генератор случайных чисел (зафиксирован seed, чтобы результаты были воспроизводимы).
    rng = np.random.default_rng(seed)

    # Инициализация популяции: равномерно в границах по каждой координате.
    population = rng.uniform(lower_bounds, upper_bounds_arr, size=(population_size, dim))

    # Выбираем FAST-окна по нагрузке (две недели "стресса": по сумме и по пику).
    windows_fast = _select_fast_windows(load_profile)

    # Кэш оценок: (wind_idx, n_wind, batt_idx, n_batt, horizon_tag, window_tag) -> (fitness, result)
    cache: Dict[Tuple[int, int, int, int, str, str], Tuple[float, OptimizationResult]] = {}

    # Счетчики фактических оценок.
    fast_evals = 0
    full_evals = 0

    # Для прогресса: общий бюджет воспринимаем как сумма.
    total_progress = max_fast_evals + max_full_evals
    last_report = 0

    # ------------------------------ Начальная оценка популяции (FAST) ------------------------------
    fitness_pop: List[float] = []  # fitness для каждого индивида популяции
    best_fast_fitness = float("inf")  # лучший найденный fitness по FAST (минимум)
    best_fast_result: Optional[OptimizationResult] = None
    best_fast_pos = population[0].copy()  # позиция лучшего индивида

    for i in range(population_size):
        if fast_evals >= max_fast_evals:
            # Если бюджет FAST исчерпан, помечаем остальных как "очень плохих".
            fitness_pop.append(float("inf"))
            continue

        fitness, result, _ = _evaluate_fast(
            load_profile,
            windows_fast,
            diesel_plants,
            hydro_plants,
            wind_turbines,
            batteries,
            wind_input_full,
            population[i],
            cache,
            n_wind_max,
            n_batt_max,
        )
        fast_evals += 1
        fitness_pop.append(fitness)

        # Обновляем лучший FAST результат (минимизация fitness).
        if fitness < best_fast_fitness:
            best_fast_fitness = fitness
            best_fast_result = result
            best_fast_pos = population[i].copy()

        last_report = _progress(progress_cb, fast_evals + full_evals, total_progress, "DE initialization", last_report)

    # Лучший FULL результат (может появиться позже при полной проверке).
    best_full_result: Optional[OptimizationResult] = None
    best_full_fitness = float("inf")

    # stagnation — счетчик стагнации по лучшему FAST fitness.
    stagnation = 0
    prev_best_fast = best_fast_fitness

    # ------------------------------ Основной цикл поколений DE ------------------------------
    for gen in range(generations):
        if fast_evals >= max_fast_evals:
            break  # не выходим за бюджет FAST

        new_population = population.copy()
        new_fitness = fitness_pop.copy()

        # Проходим по каждому индивидуу: генерируем trial и делаем селекцию.
        for i in range(population_size):
            if fast_evals >= max_fast_evals:
                break

            # Выбираем три разных индекса a,b,c (не равные i) для схемы DE/rand/1.
            idxs = list(range(population_size))
            idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)

            # "Jitter" слегка меняет mutation_factor, чтобы добавить разнообразия.
            jitter = mutation_factor * (0.9 + 0.2 * rng.random())

            # Мутант: x_a + F*(x_b - x_c)
            mutant = population[a] + jitter * (population[b] - population[c])

            # Ограничиваем мутанта границами по каждой координате.
            mutant = np.clip(mutant, lower_bounds, upper_bounds_arr)

            # Кроссовер: по маске берем координаты из mutant или из текущего population[i].
            cross_mask = rng.random(dim) < crossover_rate

            # Гарантируем хотя бы одну координату из mutant (иначе trial может полностью совпасть с родителем).
            cross_mask[rng.integers(dim)] = True

            # trial — кандидат, который будем оценивать.
            trial = np.where(cross_mask, mutant, population[i])

            # FAST-оценка trial.
            fitness_trial, result_trial, _ = _evaluate_fast(
                load_profile,
                windows_fast,
                diesel_plants,
                hydro_plants,
                wind_turbines,
                batteries,
                wind_input_full,
                trial,
                cache,
                n_wind_max,
                n_batt_max,
            )
            fast_evals += 1

            last_report = _progress(
                progress_cb,
                fast_evals + full_evals,
                total_progress,
                f"DE gen {gen + 1}/{generations}, fast={fast_evals}, full={full_evals}",
                last_report,
            )

            # Селекция: если trial не хуже текущего (fitness меньше или равно) -> заменяем.
            if fitness_trial <= fitness_pop[i]:
                new_population[i] = trial
                new_fitness[i] = fitness_trial

                # Обновляем глобально лучший FAST кандидат.
                if fitness_trial < best_fast_fitness:
                    best_fast_fitness = fitness_trial
                    best_fast_result = result_trial
                    best_fast_pos = trial.copy()

        # ------------------------------ Локальный поиск вокруг лучшего FAST ------------------------------
        eval_counter = [fast_evals]
        best_fast_pos, local_fit, local_res = _local_search(
            best_fast_pos,
            best_fast_fitness,
            load_profile,
            windows_fast,
            diesel_plants,
            hydro_plants,
            wind_turbines,
            batteries,
            wind_input_full,
            cache,
            n_wind_max,
            n_batt_max,
            rng,
            max_fast_evals,
            eval_counter,
        )
        fast_evals = eval_counter[0]

        # Если локальный поиск улучшил fitness, обновляем лучший результат.
        if local_fit < best_fast_fitness:
            best_fast_fitness = local_fit
            if local_res is not None:
                best_fast_result = local_res

        last_report = _progress(
            progress_cb,
            fast_evals + full_evals,
            total_progress,
            f"DE gen {gen + 1}/{generations} (local)",
            last_report,
        )

        # Фиксируем новое поколение.
        population = new_population
        fitness_pop = new_fitness

        # ------------------------------ Проверка стагнации ------------------------------
        if best_fast_fitness < prev_best_fast - 1e-9:
            stagnation = 0
        else:
            stagnation += 1
        prev_best_fast = best_fast_fitness

        # Если долго нет улучшения, "встряхиваем" худшие 20% популяции.
        if stagnation >= 10:
            worst_idx = np.argsort(fitness_pop)[-max(1, population_size // 5) :]
            for idx in worst_idx:
                if fast_evals >= max_fast_evals:
                    break

                # Переинициализация индивидуума случайным образом.
                population[idx] = rng.uniform(lower_bounds, upper_bounds_arr)

                # Пересчет его fitness на FAST.
                fitness_pop[idx], _, _ = _evaluate_fast(
                    load_profile,
                    windows_fast,
                    diesel_plants,
                    hydro_plants,
                    wind_turbines,
                    batteries,
                    wind_input_full,
                    population[idx],
                    cache,
                    n_wind_max,
                    n_batt_max,
                )
                fast_evals += 1
            stagnation = 0

        # ------------------------------ Плановая FULL-проверка ------------------------------
        # Раз в full_check_interval поколений делаем FULL оценку top_k уникальных дискретных решений.
        if (gen + 1) % full_check_interval == 0 and full_evals < max_full_evals:
            unique_candidates = {}

            # Сортируем популяцию по fitness (лучшие вперед) и берем top_k уникальных decoded.
            for pos, fit in sorted(zip(population, fitness_pop), key=lambda x: x[1]):
                decoded = _decode_candidate(pos, wind_turbines, batteries, n_wind_max, n_batt_max)
                if decoded not in unique_candidates:
                    unique_candidates[decoded] = pos
                if len(unique_candidates) >= top_k:
                    break

            # Проверяем каждого уникального кандидата на полном горизонте.
            for decoded, pos in unique_candidates.items():
                if full_evals >= max_full_evals:
                    break

                fitness_full, result_full, _ = _evaluate_full(
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
                )
                full_evals += 1

                last_report = _progress(
                    progress_cb,
                    fast_evals + full_evals,
                    total_progress,
                    f"DE gen {gen + 1}/{generations}, fast={fast_evals}, full={full_evals}",
                    last_report,
                )

                # Обновляем лучший FULL результат при улучшении.
                if fitness_full < best_full_fitness:
                    best_full_fitness = fitness_full
                    best_full_result = result_full

    # Если есть лучший FULL, он приоритетнее (потому что оценка точнее).
    if best_full_result is not None:
        return best_full_result

    # Если FULL не проводился/не дал улучшений, возвращаем лучший FAST результат.
    if best_fast_result is not None:
        return best_fast_result

    # Fallback: если совсем нет результата, просто считаем greedy на исходном составе.
    return greedy_optimization(
        load_profile,
        diesel_plants,
        hydro_plants,
        wind_turbines,
        batteries,
        wind_input_full,
        verbose=False,
    )
