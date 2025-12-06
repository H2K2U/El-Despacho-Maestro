# File: core/algorithms/greedy.py
"""
Реализация жадного алгоритма (dispatch) для почасовой диспетчеризации автономной энергосистемы.

1) Роль алгоритма в проекте
   Это алгоритм НИЖНЕГО уровня: при фиксированном составе оборудования (ДЭС/ДГУ, МГЭС, ВЭУ, АКБ)
   он строит почасовое расписание (schedule): сколько каждый источник выдает, куда идет энергия,
   сколько заряжается/разряжается АКБ, сколько сбрасывается (dump), и есть ли недоотпуск (unserved).

   ВАЖНО: DE/GWO (верхний уровень) вызывают greedy_optimization много раз, чтобы оценить "кандидатов состава".

2) Модель (цели/ограничения) в том виде, как она реализована тут (эвристика)
   Жадный алгоритм не решает строгую задачу линейного/нелинейного программирования,
   но фактически реализует "правила", которые приближают следующую идею:

   2.1) Приближенная целевая функция (интуитивно)
   • в первую очередь минимизировать недоотпуск нагрузки E_unserved (почти "запретить");
   • далее минимизировать расход топлива Fuel;
   • далее (вторично) минимизировать сброс E_dump и "плохие" режимы АКБ.

   2.2) Основные ограничения (выполняются процедурно, через логику шагов)
   По каждому часу t:
   • баланс мощности:
       load(t) = hydro_to_load(t) + wind_to_load(t) + diesel_to_load(t) + batt_discharge_to_load(t)
                 - batt_charge(t) - dump(t) - unserved(t)
     (в коде баланс контролируется диагностикой, а распределение делается пошагово);
   • 0 <= P_hydro(t) <= P_hydro_max(month), задается гидрографом/моделью HydroPlant;
   • 0 <= P_wind(t) <= P_wind_curve(v_wind(t)), задается кривой мощности WindTurbine;
   • ДГУ:
       для включенных агрегатов: Pmin_i <= P_i(t) <= Pmax_i;
       для выключенных: P_i(t) = 0;
     + выбирается подмножество агрегатов, покрывающее требуемую мощность;
   • АКБ:
       0 <= P_charge(t) <= P_charge_max;
       0 <= P_discharge(t) <= P_discharge_max;
       SOC_min <= SOC(t) <= SOC_max;
       SOC(t+1) = SOC(t) + eta_charge * P_charge(t) * dt - (P_discharge(t) / eta_discharge) * dt
     (это реализовано через BatteryState.request(...), где SOC ведется в кВт*ч);
   • dump(t) >= 0, unserved(t) >= 0.

3) Примечание:
   В коде зафиксировано правило: сначала используем ВИЭ (МГЭС + ВЭУ), затем АКБ, затем ДГУ,
   а при избытке энергии сначала пытаемся зарядить АКБ, затем сбрасываем в dump.
"""

from itertools import combinations
from typing import Dict, List, Tuple

import pandas as pd

# Состояние АКБ (SOC в кВт*ч и лимиты по энергии/мощности) и операции "запрос заряда/разряда".
from core.battery import BatteryState

# Диагностика энергетического баланса АКБ: проверка непротиворечивости SOC, заряд/разряд, КПД и т.д.
from core.battery_metrics import compute_battery_energy_balance

# Модели домена (оборудование, профили, результат оптимизации).
from core.models import Battery, DieselPlant, HydroPlant, LoadProfile, OptimizationResult, WindTurbine


def greedy_optimization(
    load_profile: LoadProfile,
    diesel_plants: List[DieselPlant],
    hydro_plants: List[HydroPlant],
    wind_turbines: List[WindTurbine],
    batteries: List[Battery],
    wind_speeds: List[float] = None,
    verbose: bool = False,
    **kwargs,
) -> OptimizationResult:
    """
    Жадная диспетчеризация по часам.

    Вход:
    • load_profile — профиль нагрузки (обычно DataFrame с колонкой "load" и индексом-временем);
    • diesel_plants — список дизельных станций, каждая содержит несколько ДГУ (diesel_units);
    • hydro_plants — список МГЭС (мощность зависит от месяца через available_power(month));
    • wind_turbines — каталог/список ВЭУ (в этой реализации фактически берется wind_turbines[0]);
    • batteries — каталог/список АКБ (в этой реализации фактически берется batteries[0]);
    • wind_speeds — массив скоростей ветра по часам (если задан);
    • verbose — печатать ли диагностику в консоль;
    • kwargs:
        - progress_cb: callback для прогресса (done, total, msg);
        - initial_soc_kwh: начальный SOC АКБ в кВт*ч (если хочешь задать явно).

    Выход:
    • OptimizationResult, где:
        - schedule: DataFrame с почасовыми величинами;
        - selected_equipment: структура с выбранным оборудованием;
        - total_fuel_consumption: общий расход топлива (сумма по часам);
        - renewable_energy_ratio: доля покрытия нагрузки ВИЭ (МГЭС+ВЭУ);
        - cost_analysis: пока пусто (заглушка).
    """

    # ------------------------------ 0) Подготовка служебных переменных ------------------------------

    # Сюда собираем строки расписания по часам, затем сделаем DataFrame.
    schedule_data = []

    # total_hours — количество часов моделирования (обычно 8760, но может быть меньше).
    # Важно: мы предполагаем, что load_profile.data индексируется по часам через .iloc.
    total_hours = len(load_profile.data)

    # Колбэк прогресса, если UI/внешний код хочет отображать прогресс.
    progress_cb = kwargs.get("progress_cb")

    # total_steps — сколько шагов всего (обычно равен total_hours).
    total_steps = total_hours

    # ------------------------------ 1) Выбор оборудования (упрощенно) ------------------------------
    # ВНИМАНИЕ: На данный момент выбор "упрощенный":
    # • если wind_turbines не пуст, берем НУЛЕВОЙ (первый) элемент как "выбранную ВЭУ";
    # • если batteries не пуст, берем НУЛЕВОЙ (первый) элемент как "выбранную АКБ".
    # То есть этот greedy НЕ перебирает типы оборудования, он диспетчеризует при заданном типе.
    # Перебор типов/количеств делает верхний уровень (DE/GWO), где ВЭУ/АКБ агрегируются.
    selected_wind = wind_turbines[0] if wind_turbines else None
    selected_battery = batteries[0] if batteries else None

    # ------------------------------ 2) "Плоский" список всех ДГУ ------------------------------
    # diesel_plants может содержать несколько станций, на каждой несколько агрегатов.
    # Для удобства диспетчеризации мы формируем единый список diesel_unit_entries, где каждый элемент содержит:
    # • plant: ссылка на DieselPlant;
    # • unit: ссылка на конкретный дизельный агрегат (обычно DieselUnit);
    # • name: человекочитаемое имя колонки в schedule;
    # • pmax: номинальная мощность агрегата;
    # • pmin: минимально допустимая мощность агрегата (если не задано, берем 40% от номинала).
    diesel_unit_entries = []
    unit_counter = 1
    for plant in diesel_plants:
        for unit in plant.diesel_units:
            # Если имя агрегата не задано, создаем "ДГУ 1", "ДГУ 2", ...
            unit_name = unit.name or f"ДГУ {unit_counter}"

            # Если у станции есть имя, делаем "Имя станции - Имя агрегата".
            full_name = f"{plant.name} - {unit_name}" if plant.name else unit_name

            diesel_unit_entries.append({
                "plant": plant,
                "unit": unit,
                "name": full_name,
                "pmax": unit.nominal_power,
                # pmin может быть прямо задан у unit (например, технический минимум),
                # иначе используем 0.4 * Pном как типичное приближение.
                "pmin": getattr(unit, "pmin", unit.nominal_power * 0.4),
            })
            unit_counter += 1

    # ------------------------------ 3) Инициализация состояния АКБ ------------------------------
    # battery_state — объект BatteryState, который хранит:
    # • энергию в АКБ energy_kwh (SOC в кВт*ч);
    # • границы soc_min_kwh, soc_max_kwh;
    # • лимиты мощности charge/discharge;
    # • КПД заряд/разряд;
    # • метод request(...) для "запроса" заряда/разряда с учетом всех ограничений.
    battery_state = None
    if selected_battery:
        # initial_soc_kwh можно передать через kwargs; если None, BatteryState решит сам (обычно середина диапазона).
        battery_state = BatteryState.from_battery(selected_battery, kwargs.get("initial_soc_kwh"))

    # Фиксируем стартовую энергию АКБ, чтобы потом можно было сверить баланс.
    battery_start_kwh = battery_state.energy_kwh if battery_state else None

    # Флаг, есть ли у нас вообще массив скоростей ветра.
    has_wind_speeds = wind_speeds is not None and len(wind_speeds) > 0

    # Накопитель общего расхода топлива.
    total_fuel_consumption = 0.0

    # Параметры "прогноза" (в текущей реализации почти не используется для управления,
    # но заготовка есть: оценка дефицита вперед на forecast_window часов).
    forecast_window = min(12, total_hours)  # максимум 12 часов вперед, или меньше если данных меньше
    forecast_alpha = 0.5  # сейчас не используется (возможно, планировалось сглаживание/вес)

    # =================================================================================================
    # ----------------------------------- ВНУТРЕННИЕ ФУНКЦИИ ------------------------------------------
    # =================================================================================================

    def distribute_power_among_units(required_power: float) -> Tuple[Dict[str, float], float, float]:
        """
        Подбор включаемых ДГУ и распределение мощности по агрегатам.

        required_power — сколько мощности (кВт) нужно выдать дизелем, чтобы покрыть остаток нагрузки.

        Возвращает:
        • unit_powers: Dict[str, float] — словарь "имя агрегата -> назначенная мощность (кВт)";
        • total_power: float — суммарная назначенная мощность дизеля (кВт);
        • total_pmin: float — сумма pmin включенных агрегатов (кВт).

        Идея выбора подмножества агрегатов:
        • если агрегатов мало (<= 12), перебираем все подмножества (combinatorial search) и выбираем лучшее;
        • если агрегатов много, используем упрощенный жадный выбор (по возрастанию pmin).

        Критерий выбора подмножества (лексикографически по метрикам):
        metrics = (overgen, n_units_on, fuel_consumption)
        где:
        • overgen — "перегенерация" из-за суммарного Pmin (если ΣPmin > required_power);
        • n_units_on — сколько агрегатов включили;
        • fuel_consumption — прогноз расхода топлива по нагрузке агрегатов (по unit.fuel_consumption()).

        То есть приоритет:
        1) минимизировать overgen;
        2) при равном overgen — включить меньше агрегатов;
        3) при равном — выбрать более экономичный по топливу вариант.
        """
        if required_power <= 0 or not diesel_unit_entries:
            # Если дизель не нужен или его нет, возвращаем нули по всем агрегатам.
            return {entry["name"]: 0.0 for entry in diesel_unit_entries}, 0.0, 0.0

        n_units = len(diesel_unit_entries)

        def evaluate_subset(indices: List[int]) -> Tuple[Tuple[float, int, float], Dict[str, float], float, float]:
            """
            Оценивает конкретное подмножество агрегатов (по индексам entries) и строит распределение мощности.

            Возвращает:
            • metrics: (overgen, count_selected, fuel_consumption)
            • powers: словарь мощностей для всех агрегатов (не выбранные получат 0.0)
            • total_power: суммарная мощность по всем агрегатам
            • total_pmin: суммарный минимум включенных агрегатов

            Важно:
            • если ΣPmax < required_power, подмножество не годится -> метрики = inf.
            """
            selected = [diesel_unit_entries[i] for i in indices]
            if not selected:
                return (float("inf"), float("inf"), float("inf")), {}, 0.0, 0.0

            # Проверяем достижимость: суммарный максимум должен покрыть required_power.
            total_pmax = sum(entry["pmax"] for entry in selected)
            if total_pmax + 1e-6 < required_power:
                return (float("inf"), float("inf"), float("inf")), {}, 0.0, 0.0

            # Суммарный технический минимум.
            total_pmin = sum(entry["pmin"] for entry in selected)

            # Перегенерация возникает, когда required_power меньше суммы минимумов.
            overgen = max(0.0, total_pmin - required_power)

            # powers заполняем для всех агрегатов (и включенных, и выключенных),
            # чтобы потом удобно писать в schedule колонками.
            powers = {entry["name"]: 0.0 for entry in diesel_unit_entries}

            # Базовая загрузка: каждому включенному агрегату назначаем Pmin.
            # remaining — сколько еще надо "добрать" к required_power сверх этих минимумов.
            remaining = required_power
            for entry in selected:
                powers[entry["name"]] = entry["pmin"]
                remaining -= entry["pmin"]

            # Если после назначения минимумов все еще не хватает мощности, распределяем остаток по доступному резерву.
            # Распределяем в порядке убывания "запаса" (Pmax - Pmin), чтобы быстрее закрыть требование.
            if remaining > 0:
                for entry in sorted(selected, key=lambda x: x["pmax"] - x["pmin"], reverse=True):
                    headroom = entry["pmax"] - powers[entry["name"]]  # сколько еще можно добавить этому агрегату
                    if headroom <= 0:
                        continue
                    add_power = min(headroom, remaining)
                    powers[entry["name"]] += add_power
                    remaining -= add_power
                    if remaining <= 1e-6:
                        break

            total_power = sum(powers.values())

            # Оценка расхода топлива на этом шаге (для сравнения подмножеств).
            # Здесь предполагается, что unit.fuel_consumption(load_percentage) возвращает расход топлива (например л/ч).
            fuel_consumption = 0.0
            for entry in selected:
                power = powers[entry["name"]]
                if power > 0:
                    load_percentage = min(1.0, power / entry["pmax"])  # доля нагрузки агрегата (0..1)
                    try:
                        fuel_consumption += entry["unit"].fuel_consumption(load_percentage)
                    except Exception:
                        # Если модель топлива не определена/упала, штраф не добавляем (остается 0).
                        # Это компромисс: лучше не ломать оптимизацию из-за одной кривой, но сравнение станет грубее.
                        fuel_consumption += 0.0

            metrics = (overgen, len(selected), fuel_consumption)
            return metrics, powers, total_power, total_pmin

        # Инициализация "лучшего" решения.
        best_metrics = (float("inf"), float("inf"), float("inf"))
        best_powers: Dict[str, float] = {entry["name"]: 0.0 for entry in diesel_unit_entries}
        best_total_power = 0.0
        best_total_pmin = 0.0

        # Если агрегатов немного — полный перебор всех комбинаций.
        if n_units <= 12:
            all_indices = list(range(n_units))
            for r in range(1, n_units + 1):
                for combo in combinations(all_indices, r):
                    metrics, powers, total_power, total_pmin = evaluate_subset(list(combo))
                    # Сравнение кортежей в Python лексикографическое: сначала overgen, потом кол-во агрегатов, потом топливо.
                    if metrics < best_metrics:
                        best_metrics = metrics
                        best_powers = powers
                        best_total_power = total_power
                        best_total_pmin = total_pmin
        else:
            # Если агрегатов много — жадный выбор подмножества по pmin:
            # набираем агрегаты с минимальными pmin, пока ΣPmax не покроет требуемую мощность.
            sorted_units = sorted(enumerate(diesel_unit_entries), key=lambda x: x[1]["pmin"])
            selected_indices = []
            total_pmax = 0.0
            for idx, entry in sorted_units:
                selected_indices.append(idx)
                total_pmax += entry["pmax"]
                if total_pmax >= required_power:
                    break
            best_metrics, best_powers, best_total_power, best_total_pmin = evaluate_subset(selected_indices)

        return best_powers, best_total_power, best_total_pmin

    def get_net_deficit(start_hour: int, horizon: int) -> float:
        """
        Оценка суммарного дефицита (load - (hydro + wind)) на горизонте вперед.

        start_hour — от какого часа считаем вперед;
        horizon — сколько часов вперед смотреть.

        Возвращает:
        • deficit (кВт*ч при dt=1): суммируем только положительный дефицит.

        Примечание:
        • Это "прогнозная" функция: она не использует дизель и АКБ,
          а лишь оценивает, сколько не хватит от ВИЭ.
        • В текущем коде она вычисляется всегда (net_deficit_future),
          но не влияет на управление (soc_target не меняется).
          То есть это заготовка под более умную стратегию.
        """
        window = min(horizon, total_hours - start_hour)
        deficit = 0.0
        for h in range(window):
            idx = start_hour + h

            # Нагрузка на этом часе.
            load_val = load_profile.data.iloc[idx]["load"]

            # Месяц берем из индекса DataFrame, если индекс похож на datetime.
            month_val = load_profile.data.iloc[idx].name.month if hasattr(load_profile.data.iloc[idx].name, "month") else 1

            # Доступная мощность МГЭС (сумма по всем гидроисточникам).
            hydro_val = sum(hydro.available_power(month_val) for hydro in hydro_plants)

            # Доступная мощность ВЭУ по скорости ветра.
            wind_val = 0.0
            if selected_wind and has_wind_speeds and idx < len(wind_speeds):
                try:
                    wind_val = selected_wind.available_power(wind_speeds[idx])
                except Exception:
                    wind_val = 0.0

            deficit += max(0.0, load_val - (hydro_val + wind_val))
        return deficit

    # =================================================================================================
    # ----------------------------------- ОСНОВНОЙ ПОЧАСОВОЙ ЦИКЛ -------------------------------------
    # =================================================================================================

    for hour in range(total_hours):
        # Текущая нагрузка на час hour (кВт).
        load = load_profile.data.iloc[hour]["load"]

        # Месяц нужен для гидрогенерации (обычно гидрограф по месяцам).
        month = load_profile.data.iloc[hour].name.month if hasattr(load_profile.data.iloc[hour].name, "month") else 1

        # ------------------------------ 1) МГЭС: доступная мощность на этот месяц ------------------------------
        hydro_power = 0.0
        for hydro in hydro_plants:
            hydro_power += hydro.available_power(month)

        # ------------------------------ 2) ВЭУ: мощность по скорости ветра ------------------------------
        wind_power = 0.0
        if selected_wind and has_wind_speeds and hour < len(wind_speeds):
            try:
                wind_speed = wind_speeds[hour]
                wind_power = selected_wind.available_power(wind_speed)
            except Exception as e:
                if verbose:
                    print(f"Ошибка расчета ветроэнергии на час {hour}: {e}")
                wind_power = 0.0

        # Суммарная "возобновляемая" мощность (ВИЭ): МГЭС + ВЭУ.
        renewable_power = hydro_power + wind_power

        # Сколько ВИЭ реально идет в нагрузку: не больше нагрузки, не больше выработки ВИЭ.
        renewable_to_load = min(renewable_power, load)

        # Остаток нагрузки после ВИЭ:
        # • если renewable_power < load => residual_load_after_renewables > 0 (дефицит);
        # • если renewable_power > load => residual_load_after_renewables < 0 (избыток).
        residual_load_after_renewables = load - renewable_power

        # ------------------------------ 3) Инициализация потоков/переменных текущего часа ------------------------------
        battery_discharge = 0.0                 # разряд АКБ (кВт) в этот час, который пойдет на нагрузку
        battery_charge_from_renewable = 0.0     # заряд АКБ от ВИЭ (кВт)
        battery_charge_from_diesel = 0.0        # заряд АКБ от дизеля (кВт) — в этой версии не используется (заготовка)
        diesel_power = 0.0                      # суммарная мощность дизеля (кВт)
        diesel_excess = 0.0                     # избыток дизеля относительно потребности (кВт)
        dump_power = 0.0                        # сброс/балласт (кВт)
        unserved_load = 0.0                     # недоотпуск (кВт)
        soc_before = battery_state.energy_kwh if battery_state else 0.0  # SOC до операций этого часа (кВт*ч)
        diesel_pmin_sum = 0.0                   # сумма Pmin включенных ДГУ (кВт) — полезно для анализа

        # "Прогнозный" дефицит по ВИЭ вперед (сейчас не влияет на управление).
        net_deficit_future = get_net_deficit(hour, forecast_window)

        # Целевая траектория SOC (заготовка под управление SOC по прогнозу).
        soc_target = soc_before

        # Словарь распределения мощности по агрегатам: "имя ДГУ -> мощность (кВт)".
        # Заполняем нулями для всех агрегатов, чтобы всегда были колонки в schedule.
        unit_powers: Dict[str, float] = {entry["name"]: 0.0 for entry in diesel_unit_entries}

        # ------------------------------ 4) Балансирование: дефицит (нагрузка не покрыта ВИЭ) ------------------------------
        if residual_load_after_renewables > 1e-6:
            # 4.1) Пытаемся покрыть дефицит разрядом АКБ.
            if battery_state:
                # BatteryState.request(discharge_kw=...) возвращает (действительный_заряд, действительный_разряд, ...)
                # Мы берем только величину разряда (сколько реально удалось отдать в сеть).
                _, battery_discharge, _ = battery_state.request(discharge_kw=residual_load_after_renewables)
                residual_load_after_renewables = max(0.0, residual_load_after_renewables - battery_discharge)

            # 4.2) Остаток дефицита покрываем дизелем.
            diesel_requirement = max(0.0, residual_load_after_renewables)
            if diesel_requirement > 1e-6:
                # Выбираем подмножество ДГУ и распределяем мощность.
                unit_powers, diesel_power, diesel_pmin_sum = distribute_power_among_units(diesel_requirement)

                # remaining_after_discharge — сколько еще нужно покрыть после ВИЭ + разряда АКБ.
                # Здесь используется renewable_to_load, а не renewable_power, потому что в нагрузку идет только часть ВИЭ.
                remaining_after_discharge = max(0.0, load - renewable_to_load - battery_discharge)

                # Избыток дизеля над фактической потребностью (часто появляется из-за суммарного Pmin).
                diesel_excess = max(0.0, diesel_power - remaining_after_discharge)

                # Избыток дизеля сейчас трактуем как dump (сброс).
                dump_power += max(0.0, diesel_excess)

                # Недоотпуск, если дизель все же не перекрыл потребность (например, недостаточно Pmax).
                unserved_load = max(0.0, remaining_after_discharge - diesel_power)

        # ------------------------------ 5) Балансирование: избыток ВИЭ (renewable_power > load) ------------------------------
        else:
            # renewable_excess — положительный избыток ВИЭ после покрытия нагрузки (кВт).
            renewable_excess = -residual_load_after_renewables

            # 5.1) Сначала пытаемся зарядить АКБ от избытка ВИЭ.
            if battery_state:
                battery_charge_from_renewable, _, _ = battery_state.request(charge_kw=renewable_excess)
                renewable_excess = max(0.0, renewable_excess - battery_charge_from_renewable)

            # 5.2) Остаток избытка — сбрасываем в dump.
            dump_power += max(0.0, renewable_excess)

        # ------------------------------ 6) Итог по дизелю и быстрая диагностика баланса ------------------------------
        # На всякий случай пересчитываем diesel_power как сумму поагрегатных мощностей.
        diesel_power = sum(unit_powers.values()) if diesel_unit_entries else diesel_power

        # Диагностический "баланс" (не используется в оптимизации, только предупреждение).
        # ВНИМАНИЕ: эта проверка сделана как грубый контроль; интерпретировать ее надо аккуратно,
        # потому что одновременно учитывается load и unserved как "потребление" (что может завышать расход).
        total_generation = renewable_to_load + battery_discharge + diesel_power
        total_consumption = load + battery_charge_from_renewable + battery_charge_from_diesel + dump_power + unserved_load
        balance_diff = abs(total_generation - total_consumption)
        if verbose and balance_diff > 0.1:
            print(f"Предупреждение: Дисбаланс на час {hour}: {balance_diff:.2f} кВт")

        # ------------------------------ 7) Топливо: суммируем расход топлива по каждому агрегату ------------------------------
        # entry["unit"].fuel_consumption(load_fraction) — модель расхода топлива агрегата на данном часе.
        for entry in diesel_unit_entries:
            power = unit_powers.get(entry["name"], 0.0)
            if power > 0:
                try:
                    total_fuel_consumption += entry["unit"].fuel_consumption(power / entry["pmax"])
                except Exception:
                    total_fuel_consumption += 0.0

        # SOC после всех операций часа (внутри battery_state энергия уже обновлена).
        soc_after = battery_state.energy_kwh if battery_state else soc_before

        # ------------------------------ 8) Формирование строки расписания (schedule_row) ------------------------------
        # Сюда складываем все ключевые поля, которые потом используют:
        # • визуализация;
        # • метрики;
        # • верхние оптимизаторы (DE/GWO) при подсчете fitness.
        schedule_row = {
            "hour": hour,
            "load": load,

            # Источники ВИЭ
            "hydro": hydro_power,
            "wind": wind_power,

            # Сколько ВИЭ реально пошло в нагрузку
            "renewable_to_load": renewable_to_load,

            # Разряд АКБ
            "battery_discharge": battery_discharge,
            "battery_discharge_to_load": battery_discharge,

            # Заряд АКБ от ВИЭ (в проекте встречаются разные ключи — оставлены оба для совместимости)
            "battery_charge_from_renewable": battery_charge_from_renewable,
            "battery_charge_from_RES": battery_charge_from_renewable,

            # Заряд АКБ от дизеля (заготовка; сейчас всегда 0.0)
            "battery_charge_from_diesel": battery_charge_from_diesel,
            "battery_charge_from_DIESEL": battery_charge_from_diesel,

            # Дизель (суммарно)
            "diesel": diesel_power,

            # Сброс (dump) и недоотпуск (unserved)
            "dump": dump_power,
            "unserved": unserved_load,

            # Избыток дизеля (часто из-за суммарного Pmin)
            "diesel_excess": diesel_excess,

            # SOC в кВт*ч (в проекте используются оба названия)
            "soc": soc_after,
            "soc_kwh": soc_after,

            # Целевой SOC (заготовка), сейчас равен soc_before
            "soc_target": soc_target,

            # Для совместимости с метриками/визуализацией:
            # общая мощность заряда и разряда в этот час.
            "charge_power": battery_charge_from_renewable + battery_charge_from_diesel,
            "discharge_power": battery_discharge,
        }

        # Добавляем поагрегатные мощности ДГУ как отдельные колонки в schedule.
        schedule_row.update(unit_powers)

        schedule_data.append(schedule_row)

        # Репортим прогресс раз в 25 часов (или на последнем шаге), чтобы не перегружать UI.
        if progress_cb and (hour % 25 == 0 or hour == total_steps - 1):
            progress_cb(hour + 1, total_steps, f"час {hour + 1}/{total_steps}")

    # Превращаем список dict'ов в таблицу.
    schedule_df = pd.DataFrame(schedule_data)

    # ------------------------------ 9) Базовые итоговые метрики по ВИЭ ------------------------------
    renewable_energy = schedule_df["renewable_to_load"].sum()
    total_energy = schedule_df["load"].sum()

    # ------------------------------ 10) Диагностика по первым суткам/двум суткам (если verbose) ------------------------------
    # Это "трассировка" для выявления проблем типа:
    # • расхождение SOC start/end;
    # • слишком большой заряд/разряд;
    # • наличие unserved/dump в коротком фрагменте.
    if verbose and len(schedule_df) >= 24:
        trace = schedule_df.head(min(48, len(schedule_df)))
        soc_start_trace = float(trace.iloc[0]["soc"]) if not trace.empty else 0.0
        soc_end_trace = float(trace.iloc[-1]["soc"]) if not trace.empty else 0.0
        total_charge = float(trace["charge_power"].sum()) if "charge_power" in trace else 0.0
        total_discharge = float(trace["discharge_power"].sum()) if "discharge_power" in trace else 0.0
        print(
            "Trace 2 days =>",
            f"soc_start={soc_start_trace:.4f} kWh",
            f"soc_end={soc_end_trace:.4f} kWh",
            f"delta={soc_end_trace - soc_start_trace:.4f} kWh",
            f"charge={total_charge:.4f} kW",
            f"discharge={total_discharge:.4f} kW",
            f"unserved={float(trace['unserved'].sum()):.4f} kWh",
            f"dump={float(trace['dump'].sum()):.4f} kWh",
        )

    # ------------------------------ 11) Проверка: общий недоотпуск ------------------------------
    total_unserved = schedule_df["unserved"].sum()
    if verbose and total_unserved > 0.1:
        print(f"ВНИМАНИЕ: Общая непокрытая нагрузка: {total_unserved:.2f} кВт*ч")

    # ------------------------------ 12) Проверка: одновременный заряд и разряд АКБ ------------------------------
    # Это типовой "антипаттерн": в физически корректной модели он должен быть запрещен,
    # либо должен быть крайне мал (на уровне численной погрешности).
    simultaneous_charge_discharge = 0
    for _, hour_data in schedule_df.iterrows():
        if hour_data.get("battery_discharge", 0) > 0.1 and (
            hour_data.get("battery_charge_from_renewable", 0) > 0.1
            or hour_data.get("battery_charge_from_diesel", 0) > 0.1
        ):
            simultaneous_charge_discharge += 1

    if verbose and simultaneous_charge_discharge > 0:
        print(f"ВНИМАНИЕ: Обнаружено одновременных зарядов/разрядов АКБ: {simultaneous_charge_discharge} часов")

    # ------------------------------ 13) Метаданные АКБ + расширенная диагностика баланса ------------------------------
    # battery_meta — структура, которую удобно отображать в UI и передавать в battery_metrics.
    battery_meta = {
        "name": selected_battery.name if selected_battery else None,

        # Емкость одной выбранной АКБ (или агрегированной АКБ, если сюда пришел агрегированный объект)
        "capacity": selected_battery.capacity if selected_battery else None,
        "capacity_total": selected_battery.capacity if selected_battery else None,

        # Предельные мощности
        "max_charge_power": selected_battery.max_charge_power if selected_battery else None,
        "max_discharge_power": selected_battery.max_discharge_power if selected_battery else None,

        # Границы SOC в кВт*ч, рассчитанные BatteryState (важно: soc_min/soc_max в Battery могут быть долями, BatteryState переводит)
        "soc_min_kwh": battery_state.soc_min_kwh if battery_state else None,
        "soc_max_kwh": battery_state.soc_max_kwh if battery_state else None,

        # КПД заряд/разряд (если в BatteryState они раздельные)
        "eta_charge": battery_state.eta_charge if battery_state else None,
        "eta_discharge": battery_state.eta_discharge if battery_state else None,

        # Старт/финиш энергии (SOC в кВт*ч)
        "soc_start_kwh": battery_start_kwh,
        "soc_end_kwh": battery_state.energy_kwh if battery_state else None,

        # Источник стартового SOC (для логирования/диагностики)
        "soc_start_source": "initial",
    }

    # Если АКБ есть, считаем энергетический баланс и добавляем результат в battery_meta.
    # Это помогает ловить ошибки типа "SOC на следующий день не равен SOC на конец предыдущего".
    if battery_state:
        balance = compute_battery_energy_balance(schedule_df, battery_meta, soc_start_override=battery_start_kwh)
        battery_meta.update(balance)

    # ------------------------------ 14) selected_equipment_safe: компактное описание выбранного оборудования ------------------------------
    # Это то, что UI может показывать как "итоговый состав".
    selected_equipment_safe = {
        "wind_turbines": [{
            "name": selected_wind.name,
            "nominal_power": selected_wind.nominal_power,
            "height": selected_wind.height,
        }] if selected_wind else [],

        "batteries": [battery_meta] if selected_battery else [],

        "hydro_plants": [{
            "name": hydro.name,
            "nominal_power": hydro.nominal_power,
            "efficiency": hydro.efficiency,
        } for hydro in hydro_plants],

        "diesel_plants": [{
            "name": plant.name,
            "total_capacity": plant.total_capacity,
            "num_units": len(plant.diesel_units),
        } for plant in diesel_plants],
    }

    # ------------------------------ 15) Возврат результата ------------------------------
    return OptimizationResult(
        schedule=schedule_df,
        selected_equipment=selected_equipment_safe,
        total_fuel_consumption=total_fuel_consumption,
        renewable_energy_ratio=renewable_energy / total_energy if total_energy > 0 else 0,
        cost_analysis={},  # заглушка: сюда можно потом добавить LCOE/NPV/opex/capex и т.д.
    )
