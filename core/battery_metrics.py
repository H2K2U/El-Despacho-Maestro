# File: core/battery_metrics.py
"""
Общие утилиты для расчета "энергетического баланса АКБ" (баланса SOC).

Зачем этот модуль нужен в проекте:
• Он считает метрики баланса энергии АКБ по данным расписания (schedule DataFrame),
  причем делает это единообразно и без привязки к UI.
• Его можно использовать и в тестах (backend), и в интерфейсе (GUI), чтобы не было ситуации:
  "в тестах один баланс, в UI другой".

Ключевая договоренность (конвенция) проекта по балансу SOC:

- total_charge_kwh — прирост SOC со стороны батареи (SOC-side) от заряда:
  sum(P_charge_kw * eta_charge * dt_hours)

- total_discharge_kwh — уменьшение SOC со стороны батареи (SOC-side) от разряда:
  sum(P_discharge_kw / eta_discharge * dt_hours)

- delta_soc_expected_kwh = total_charge_kwh - total_discharge_kwh

Резидент (residual) показывает несовпадение:
• delta_soc_actual (по данным soc в таблице) И delta_soc_expected (по мощности заряда/разряда + КПД).
Если residual выходит за допуски — добавляется предупреждение.

Важно:
• Баланс и предупреждения вычисляются по НЕокругленным значениям; округления — только для отображения.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def compute_charge_discharge(day_data: pd.DataFrame) -> Dict:
    """
    Возвращает массивы заряда и разряда (кВт) как неотрицательные массивы + метаданные "откуда взяли".

    На вход:
    • day_data — DataFrame (обычно это schedule или его нарезка по суткам/окну).

    Логика выбора столбцов (чтобы не было двойного учета):

    1) Если в таблице есть агрегированные столбцы:
       • charge_power — общий заряд (кВт)
       • discharge_power — общий разряд (кВт)
       Тогда мы ПРЕДПОЧИТАЕМ их, потому что:
       • они уже агрегируют все источники;
       • если одновременно есть и агрегированные, и детальные (battery_charge_from_...),
         то суммировать все — значит посчитать заряд/разряд дважды.

    2) Если агрегированных нет:
       • собираем заряд из детальных компонент:
         - battery_charge_from_renewable и его синоним battery_charge_from_RES
         - battery_charge_from_diesel и его синоним battery_charge_from_DIESEL
       • собираем разряд из одного из столбцов:
         - battery_discharge_to_load или battery_discharge
         Берем первый найденный (опять же чтобы не удвоить).

    Возвращаемая структура:
    {
      "charge_kw": np.ndarray,      # длиной n_rows, заряд в кВт (>=0)
      "discharge_kw": np.ndarray,   # длиной n_rows, разряд в кВт (>=0)
      "metadata": {...}            # какие столбцы использовались и почему
    }

    Исключения:
    • если не найдено НИ одного адекватного столбца заряда/разряда — ValueError.
    """

    if day_data is None:
        raise ValueError("day_data must be a DataFrame")

    n_rows = len(day_data)
    charge_kw = np.zeros(n_rows)     # сюда положим итоговый заряд (кВт) по строкам
    discharge_kw = np.zeros(n_rows)  # сюда положим итоговый разряд (кВт) по строкам

    # charge_components — если будем собирать заряд из нескольких компонент,
    # тут сохраним вклад каждой компоненты (например, раздельно заряд от ВИЭ и от ДЭС).
    charge_components: Dict[str, np.ndarray] = {}

    # Списки "какие столбцы мы реально использовали"
    charge_columns_used: list[str] = []
    discharge_columns_used: list[str] = []

    # Флаги наличия агрегированных столбцов
    aggregated_charge_used = "charge_power" in day_data.columns
    aggregated_discharge_used = "discharge_power" in day_data.columns

    # --- 1) Агрегированный заряд ---
    if aggregated_charge_used:
        # np.clip(..., 0, None) обрезает отрицательные значения ниже 0:
        # заряд не должен быть отрицательным.
        charge_kw = np.clip(day_data["charge_power"].to_numpy(), 0.0, None)
        charge_columns_used.append("charge_power")

    # --- 2) Агрегированный разряд ---
    if aggregated_discharge_used:
        discharge_kw = np.clip(day_data["discharge_power"].to_numpy(), 0.0, None)
        discharge_columns_used.append("discharge_power")

    def _pick_component(columns: Iterable[str]) -> Tuple[str | None, np.ndarray]:
        """
        Внутренняя утилита: выбирает "один лучший" столбец из списка синонимов.

        Аргументы:
        • columns — набор возможных названий столбца-синонима (например:
          ["battery_charge_from_renewable", "battery_charge_from_RES"]).

        Возвращает:
        • (selected_col, selected_series)
          где selected_col — имя выбранного столбца (или None),
              selected_series — массив кВт (>=0) длиной n_rows.

        Почему так устроено:
        • если в data есть сразу два синонима, которые содержат одно и то же,
          суммировать их нельзя (двойной учет);
        • поэтому — выбираем ПЕРВЫЙ подходящий столбец.
        • а еще есть попытка выбрать "ненулевой" столбец, если первый оказался весь нулевой.
        """
        selected_col = None
        selected_series = np.zeros(n_rows)

        for col in columns:
            if col in day_data.columns:
                series = np.clip(day_data[col].to_numpy(), 0.0, None)

                if selected_col is None:
                    # первый найденный столбец берем как кандидат
                    selected_col = col
                    selected_series = series
                else:
                    # если второй столбец реально содержит ненули,
                    # а выбранный ранее столбец был полностью нулевой — перекидываемся.
                    if np.any(series > 0) and not np.any(selected_series > 0):
                        selected_col = col
                        selected_series = series

                # если нашли столбец, который вообще имеет ненулевые значения —
                # прерываем поиск: это достаточно хороший выбор.
                if np.any(series > 0):
                    break

        return selected_col, selected_series

    # --- 3) Если нет агрегированного заряда — собираем по детальным компонентам ---
    if not aggregated_charge_used:
        renewable_col, renewable_series = _pick_component(
            ["battery_charge_from_renewable", "battery_charge_from_RES"]
        )
        diesel_col, diesel_series = _pick_component(
            ["battery_charge_from_diesel", "battery_charge_from_DIESEL"]
        )

        # Сохраняем компоненты и имена использованных столбцов
        if renewable_col:
            charge_components[renewable_col] = renewable_series
            charge_columns_used.append(renewable_col)
        if diesel_col:
            charge_components[diesel_col] = diesel_series
            charge_columns_used.append(diesel_col)

        # Если нашли хотя бы одну компоненту — итоговый заряд = сумма компонент.
        if charge_components:
            charge_kw = sum(charge_components.values())

    # --- 4) Если нет агрегированного разряда — берем первый подходящий детальный столбец ---
    if not aggregated_discharge_used:
        discharge_candidates = ["battery_discharge_to_load", "battery_discharge"]
        for col in discharge_candidates:
            if col in day_data.columns:
                discharge_kw = np.clip(day_data[col].to_numpy(), 0.0, None)
                discharge_columns_used.append(col)
                # принцип "первый найденный" — чтобы не удваивать одно и то же
                break

    # Если не удалось найти вообще ничего — значит schedule не содержит данных по АКБ
    if not charge_columns_used and not discharge_columns_used:
        raise ValueError("В данных отсутствуют столбцы заряда/разряда АКБ")

    metadata = {
        # Источник данных: aggregated / detailed
        "charge_source": "aggregated" if aggregated_charge_used else "detailed",
        "discharge_source": "aggregated" if aggregated_discharge_used else "detailed",
        # Какие конкретно колонки реально использовались
        "charge_columns_used": charge_columns_used,
        "discharge_columns_used": discharge_columns_used,
        # Детализация компонент зарядов (если применимо)
        "charge_components": charge_components,
    }

    return {"charge_kw": charge_kw, "discharge_kw": discharge_kw, "metadata": metadata}


def _extract_power_arrays(day_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вспомогательная функция для обратной совместимости.

    Возвращает:
    • (charge_kw_array, discharge_kw_array)

    Это "тонкая" обертка над compute_charge_discharge, чтобы старый код
    мог продолжать работать, ожидая именно tuple массивов.
    """
    result = compute_charge_discharge(day_data)
    return result["charge_kw"], result["discharge_kw"]


def _infer_dt_hours(index: pd.Index, default: float = 1.0) -> np.ndarray:
    """
    Определяет длительность одного шага dt (в часах) по индексу DataFrame.

    Аргументы:
    • index — индекс DataFrame (обычно day_data.index);
    • default — запасной вариант (по умолчанию 1 час), если dt нельзя надежно вывести.

    Логика:
    • если индекс — DatetimeIndex и в нем > 1 точки:
      - берем разности между соседними временными отметками;
      - переводим секунды в часы;
      - берем median (медиану) как наиболее устойчивую оценку dt;
      - возвращаем массив длины len(index), заполненный этим dt.
    • иначе — возвращаем массив длины len(index), заполненный default.

    Почему медиана:
    • если вдруг есть редкие сбои/пропуски/двойные шаги, медиана устойчивее среднего.
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) > 1:
        diffs = index.to_series().diff().dt.total_seconds().iloc[1:] / 3600.0
        if diffs.empty:
            return np.full(len(index), default)

        dt = diffs.median()
        if not np.isfinite(dt) or dt <= 0:
            dt = default

        return np.full(len(index), dt)

    return np.full(len(index), default)


def compute_battery_energy_balance(
    day_data: pd.DataFrame,
    battery_meta: Dict,
    *,
    soc_start_override: float | None = None,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-3,
) -> Dict:
    """
    Считает метрики баланса энергии АКБ по срезу расписания (schedule).

    Параметры:
    • day_data — DataFrame с расписанием (может быть сутки, окно FAST, весь год — неважно);
    • battery_meta — метаданные батареи (обычно ты формируешь их в greedy и кладешь в selected_equipment),
      ожидаемые ключи:
      - eta_charge, eta_discharge (если нет — считаем 1.0);
      - soc_start_kwh (если нет soc в таблице и нет override);
      - soc_end_kwh (опционально как запасной вариант);
      - warnings (список строк, если уже есть предупреждения).
    • soc_start_override — если задано, принудительно считаем soc_start_kwh = это значение;
      полезно когда "старт SOC" нужно задать явно (например, старт года).
    • rel_tol, abs_tol — допуски на невязку residual:
      - tolerance = max(abs_tol, rel_tol * abs(delta_soc_expected))

    Что именно считается (SOC-side):
    1) Вынимаем массивы мощностей по заряду/разряду (кВт) через compute_charge_discharge().
    2) Вычисляем dt_hours на каждый шаг по индексу (обычно 1 час).
    3) Считаем:
       • charge_soc_kwh = sum(P_charge * eta_charge * dt)
       • discharge_soc_kwh = sum(P_discharge / eta_discharge * dt)
       • delta_soc_expected = charge_soc_kwh - discharge_soc_kwh
    4) Достаем фактический SOC из таблицы (если есть столбец soc или soc_kwh):
       • soc_start = soc_series[0], soc_end = soc_series[-1]
       иначе берем из battery_meta (или override).
    5) delta_soc_actual = soc_end - soc_start
    6) residual = delta_soc_actual - delta_soc_expected
       Если |residual| > tolerance — добавляем предупреждение.

    Возвращает dict с метриками, чтобы можно было:
    • показать в отчете/GUI;
    • валидировать тестами;
    • логировать качество баланса.
    """

    # 1) Берем мощности заряда/разряда (кВт) из schedule
    charge_discharge = compute_charge_discharge(day_data)
    charge_power, discharge_power = charge_discharge["charge_kw"], charge_discharge["discharge_kw"]

    # 2) КПД: если eta_discharge не задан — считаем равным eta_charge.
    eta_charge = float(battery_meta.get("eta_charge", 1.0) or 1.0)
    eta_discharge = float(battery_meta.get("eta_discharge", eta_charge) or eta_charge)

    # 3) Оценка dt по индексу (обычно 1 час).
    dt_hours = _infer_dt_hours(day_data.index)

    # 4) SOC-side энергия заряда: то, насколько SOC должен вырасти из-за зарядов.
    charge_soc_kwh = float(np.sum(charge_power * eta_charge * dt_hours))

    # 5) SOC-side энергия разряда: то, насколько SOC должен уменьшиться из-за разрядов.
    # discharge_power — полезная мощность в нагрузку, поэтому делим на eta_discharge.
    discharge_soc_kwh = float(np.sum(discharge_power / max(eta_discharge, 1e-9) * dt_hours))

    # 6) Ожидаемое изменение SOC по мощностям
    delta_soc_expected = charge_soc_kwh - discharge_soc_kwh

    # 7) Пытаемся найти фактический SOC ряд в данных
    soc_series = None
    if "soc" in day_data.columns:
        soc_series = day_data["soc"].to_numpy()
    elif "soc_kwh" in day_data.columns:
        soc_series = day_data["soc_kwh"].to_numpy()

    # 8) Определяем soc_start и soc_end
    soc_start = soc_start_override
    if soc_series is not None:
        # Если SOC есть в таблице — это наиболее надежный источник.
        if soc_start is None:
            soc_start = float(soc_series[0])
        soc_end = float(soc_series[-1])
    else:
        # Если SOC в таблице нет — берем из метаданных батареи.
        if soc_start is None:
            soc_start = float(battery_meta.get("soc_start_kwh", 0.0) or 0.0)
        soc_end = float(battery_meta.get("soc_end_kwh", soc_start))

    # 9) Фактическое изменение SOC "по данным"
    delta_soc_actual = soc_end - soc_start

    # 10) Невязка: насколько "факт SOC" расходится с "ожидаемым SOC" от заряд/разряд
    residual = delta_soc_actual - delta_soc_expected

    # Допуск: либо абсолютный, либо относительный от ожидаемого изменения SOC (что больше)
    tolerance = max(abs_tol, rel_tol * abs(delta_soc_expected))

    # 11) Предупреждения: если residual слишком большой — фиксируем
    warnings: Iterable[str] = battery_meta.get("warnings", []) or []
    warnings = list(warnings)

    if abs(residual) > tolerance:
        warnings.append(
            # Сообщение сейчас на английском — переводим на русский (как ты просил).
            "несовпадение баланса энергии АКБ: residual={:.6f} кВт*ч (soc_start={:.6f}, "
            "soc_end={:.6f}, charge_soc={:.6f}, discharge_soc={:.6f}, "
            "eta_charge={:.4f}, eta_discharge={:.4f})".format(
                residual,
                soc_start,
                soc_end,
                charge_soc_kwh,
                discharge_soc_kwh,
                eta_charge,
                eta_discharge,
            )
        )

    return {
        # Фактические границы SOC на выбранном срезе
        "soc_start_kwh": soc_start,
        "soc_end_kwh": soc_end,

        # Фактическое изменение SOC (по данным soc)
        "delta_soc_kwh": delta_soc_actual,

        # "Сколько SOC должно было измениться" от интеграла заряд/разряд (SOC-side)
        "total_charge_kwh": charge_soc_kwh,
        "total_discharge_kwh": discharge_soc_kwh,
        "delta_soc_expected_kwh": delta_soc_expected,

        # Невязка баланса (факт - ожидание)
        "energy_balance_residual_kwh": residual,

        # Список предупреждений (включая, возможно, новые)
        "warnings": warnings,
    }
