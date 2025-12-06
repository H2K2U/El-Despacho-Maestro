# File: core/battery.py
"""
Состояние АКБ (SOC) и утилиты для "шагового" расчета заряда/разряда.

Назначение модуля:
• собрать в одном месте всю математику по АКБ, чтобы ВСЕ алгоритмы (greedy / DE / GWO и т.д.)
  обращались к батарее одинаково и не расходились по SOC из-за разных трактовок КПД/границ/округлений;
• работать строго в единицах:
  ― энергия: кВт*ч (kWh);
  ― мощность: кВт (kW);
  ― шаг расчета: часы (dt_hours).

Почему это важно:
• если каждый алгоритм будет по-своему считать "сколько энергии ушло при разряде" или "как учитывать КПД",
  SOC почти гарантированно начнет "плыть" и расходиться между графиками;
• здесь специально используется детерминированное округление (_round_energy), чтобы уменьшить дрейф float.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional, Tuple


def _round_energy(value: float, ndigits: int = 6) -> float:
    """
    Детерминированное округление энергии, чтобы уменьшить накопление ошибок float.

    Аргументы:
    • value — значение энергии (кВт*ч), которое хотим "стабилизировать";
    • ndigits — число знаков после запятой, по умолчанию 6.

    Почему это нужно:
    • при 8760 шагах (год по часам) маленькие ошибки double могут накапливаться;
    • а потом появляются "микро-дисбалансы" и различия SOC "цифра в цифру";
    • round(...) делает траекторию SOC воспроизводимой и более стабильной.

    Важно:
    • округление — компромисс. Слишком грубое ухудшит точность, слишком мелкое не даст эффекта.
    """
    return round(value, ndigits)


@dataclass
class BatteryState:
    """
    ДИНАМИЧЕСКОЕ (runtime) состояние АКБ в единицах кВт*ч и кВт.

    Это не "паспорт" батареи, а именно состояние, которое меняется по шагам времени.

    Поля:
    • capacity_kwh — номинальная емкость (кВт*ч). В расчетах здесь используется в основном для
      преобразования долей SOC в кВт*ч, если SOC задан как 0..1;
    • soc_min_kwh — нижняя граница энергии (кВт*ч), ниже нее разряжать нельзя;
    • soc_max_kwh — верхняя граница энергии (кВт*ч), выше нее заряжать нельзя;
    • max_charge_kw — максимальная мощность заряда (кВт);
    • max_discharge_kw — максимальная мощность разряда (кВт);
    • eta_charge — КПД "в одну сторону" при заряде (0..1);
    • eta_discharge — КПД "в одну сторону" при разряде (0..1);
    • energy_kwh — текущая энергия в батарее (кВт*ч), т.е. текущий SOC в абсолютных единицах.

    Принципиальный момент по КПД:
    • В проекте может быть задана одна эффективность battery.efficiency как round-trip (туда-обратно).
      Тогда здесь она разбивается на две однонаправленные эффективности:
      eta_charge ≈ eta_discharge ≈ sqrt(eta_rt).
    """

    capacity_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_charge: float
    eta_discharge: float
    energy_kwh: float

    @classmethod
    def from_battery(
        cls,
        battery,
        initial_soc_kwh: Optional[float] = None,
    ) -> "BatteryState":
        """
        Создает BatteryState из "статической спецификации" батареи (объекта battery из core.models).

        Аргументы:
        • battery — объект, у которого ожидаются поля:
          - capacity (кВт*ч);
          - max_charge_power, max_discharge_power (кВт);
          - soc_min, soc_max (либо доли 0..1, либо уже кВт*ч — см. ниже);
          - efficiency (round-trip), либо eta_charge/eta_discharge (односторонние).
        • initial_soc_kwh — начальная энергия в АКБ (кВт*ч); если None, ставим середину между min и max.

        Логика по КПД:
        1) Если у батареи НЕ заданы eta_charge и eta_discharge:
           • берем eta_rt = battery.efficiency (если нет — 0.95);
           • делим round-trip на две одинаковые части: sqrt(eta_rt).
        2) Если задана только одна из eta_charge/eta_discharge:
           • вторую приравниваем к заданной (симметричное допущение).
        3) Если задан и eta_rt, и eta_charge==eta_discharge==eta_rt:
           • считаем, что eta_rt на самом деле тоже round-trip, и снова делим через sqrt.

        Логика по границам SOC:
        • battery.soc_min и battery.soc_max могут быть:
          - долями (0..1): тогда переводим в кВт*ч как soc * capacity;
          - абсолютами (кВт*ч): если значения > 1.0, трактуем их как кВт*ч.
        """

        # -------------- КПД: собираем значения из объекта battery --------------
        eta_charge = getattr(battery, "eta_charge", None)
        eta_discharge = getattr(battery, "eta_discharge", None)
        eta_rt = getattr(battery, "efficiency", None)

        # Случай 1: нет односторонних КПД — делим round-trip на две части.
        if eta_charge is None and eta_discharge is None:
            eta_rt = eta_rt if eta_rt is not None else 0.95
            eta_split = sqrt(max(eta_rt, 1e-6))  # защита от sqrt(0) и sqrt(отрицательного)
            eta_charge = eta_split
            eta_discharge = eta_split
        else:
            # Случай 2: задан только один — второй копируем.
            if eta_charge is None:
                eta_charge = eta_discharge
            if eta_discharge is None:
                eta_discharge = eta_charge

            # Случай 3: если кто-то записал eta_rt как однонаправленный (по ошибке),
            # а на самом деле это round-trip, выявляем по условию eta_charge==eta_discharge==eta_rt
            # и тоже делим через sqrt.
            if eta_rt is not None and eta_charge == eta_discharge == eta_rt:
                eta_split = sqrt(max(eta_rt, 1e-6))
                eta_charge = eta_split
                eta_discharge = eta_split

        # -------------- Границы SOC: берем из battery, задаем дефолты --------------
        soc_min = getattr(battery, "soc_min", 0.0) or 0.0
        soc_max = getattr(battery, "soc_max", 1.0) or 1.0

        # capacity в кВт*ч (может отсутствовать или быть None).
        capacity_kwh = float(getattr(battery, "capacity", 0.0) or 0.0)

        # Если емкость известна, то:
        # • если soc_min <= 1.0 — трактуем как долю и переводим в кВт*ч;
        # • иначе soc_min уже в кВт*ч.
        # Аналогично для soc_max.
        #
        # Если емкость нулевая/неизвестна — просто берем как есть.
        if capacity_kwh > 0:
            soc_min_kwh = soc_min * capacity_kwh if soc_min <= 1.0 else soc_min
            soc_max_kwh = soc_max * capacity_kwh if soc_max <= 1.0 else soc_max
        else:
            soc_min_kwh = soc_min
            soc_max_kwh = soc_max

        # -------------- Начальный SOC в кВт*ч --------------
        # Если initial_soc_kwh не задан:
        # • ставим середину допустимого диапазона [soc_min_kwh, soc_max_kwh]
        if initial_soc_kwh is None:
            energy_kwh = 0.5 * (soc_min_kwh + soc_max_kwh)
        else:
            energy_kwh = initial_soc_kwh

        # Гарантируем, что стартовая энергия в пределах допустимого диапазона,
        # и округляем для детерминизма.
        energy_kwh = _round_energy(max(soc_min_kwh, min(soc_max_kwh, energy_kwh)))

        # -------------- Создаем и возвращаем BatteryState --------------
        return cls(
            capacity_kwh=capacity_kwh,
            soc_min_kwh=soc_min_kwh,
            soc_max_kwh=soc_max_kwh,
            max_charge_kw=float(getattr(battery, "max_charge_power", 0.0) or 0.0),
            max_discharge_kw=float(getattr(battery, "max_discharge_power", 0.0) or 0.0),
            eta_charge=float(eta_charge),
            eta_discharge=float(eta_discharge),
            energy_kwh=energy_kwh,
        )

    def clone(self) -> "BatteryState":
        """
        Создает "копию" состояния батареи.

        Зачем:
        • иногда нужно попробовать "виртуальный" шаг/окно, не разрушая оригинальный SOC;
        • или передать состояние в подфункцию, чтобы она могла менять SOC локально.

        Здесь копируются ВСЕ параметры и текущая энергия.
        """
        return BatteryState(
            capacity_kwh=self.capacity_kwh,
            soc_min_kwh=self.soc_min_kwh,
            soc_max_kwh=self.soc_max_kwh,
            max_charge_kw=self.max_charge_kw,
            max_discharge_kw=self.max_discharge_kw,
            eta_charge=self.eta_charge,
            eta_discharge=self.eta_discharge,
            energy_kwh=self.energy_kwh,
        )

    def request(
        self, charge_kw: float = 0.0, discharge_kw: float = 0.0, dt_hours: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Применяет 1 шаг заряда/разряда, обновляет self.energy_kwh и возвращает "факт" выполнения.

        Аргументы:
        • charge_kw — запрошенная мощность заряда (кВт), должна быть >= 0;
        • discharge_kw — запрошенная мощность разряда (кВт), это мощность, которую хотим ОТДАТЬ В НАГРУЗКУ;
        • dt_hours — длительность шага (часы), по умолчанию 1 час.

        Возвращает:
        • actual_charge_kw — фактически выполненный заряд (кВт);
        • actual_discharge_kw — фактически выполненный разряд (кВт);
        • energy_after_kwh — энергия в батарее после шага (кВт*ч).

        Ключевые детали реализации:

        (A) Запрет одновременного заряда и разряда
            Если одновременно charge_kw > 0 и discharge_kw > 0:
            • оставляем доминирующее действие (большее по мощности),
              второе принудительно обнуляем.
            Это важно для реализма и для предотвращения "внутренней круговой"
            перекачки энергии, которая портит баланс.

        (B) Ограничения по мощности и по SOC
            Заряд ограничивается:
            • max_charge_kw — паспортное ограничение по мощности;
            • свободным объемом до soc_max_kwh и КПД заряда.

            Разряд ограничивается:
            • max_discharge_kw — паспортное ограничение по мощности;
            • запасом энергии над soc_min_kwh и КПД разряда.

        (C) Как учитывается КПД
            Здесь используется следующая трактовка:

            • При заряде:
              если мы "заливаем" в батарею мощность charge_kw на dt:
              прирост энергии батареи = charge_kw * eta_charge * dt

            • При разряде:
              discharge_kw — это полезная мощность, отданная в нагрузку.
              Тогда батарея должна "потратить" больше энергии из-за потерь:
              расход энергии батареи = discharge_kw / eta_discharge * dt

            То есть формула приращения энергии:
              delta_kwh = charge_kw * eta_charge * dt - (discharge_kw / eta_discharge) * dt

        (D) Защита от деления на ноль
            Везде, где есть деление на eta или dt, стоит max(..., 1e-9),
            чтобы при странных входных значениях не получить ZeroDivisionError.
        """

        # Приводим входы к корректным типам и исключаем отрицательные значения.
        charge_kw = max(0.0, float(charge_kw))
        discharge_kw = max(0.0, float(discharge_kw))

        # (A) Запрещаем одновременно заряжать и разряжать:
        # сохраняем только "главное" действие.
        if charge_kw > 0 and discharge_kw > 0:
            if charge_kw >= discharge_kw:
                discharge_kw = 0.0
            else:
                charge_kw = 0.0

        # ----------------- Путь заряда -----------------
        # Сначала ограничиваем паспортной мощностью.
        charge_kw = min(charge_kw, self.max_charge_kw)

        if self.soc_max_kwh > self.energy_kwh:
            # Сколько энергии МЫ ЕЩЕ МОЖЕМ ДОБАВИТЬ в батарею (внутренне),
            # но учитывая КПД заряда: чтобы увеличить энергию батареи на ΔE,
            # нужно подать ΔE / eta_charge "входной" энергии.
            charge_energy_possible = (self.soc_max_kwh - self.energy_kwh) / max(self.eta_charge, 1e-9)

            # Переводим возможную энергию в допустимую мощность на шаге dt_hours:
            # P_max_by_soc = (E_possible / dt).
            charge_kw = min(charge_kw, charge_energy_possible / max(dt_hours, 1e-9))
        else:
            # Если батарея уже на верхней границе — заряд запрещен.
            charge_kw = 0.0

        # ----------------- Путь разряда -----------------
        # Сначала ограничиваем паспортной мощностью.
        discharge_kw = min(discharge_kw, self.max_discharge_kw)

        if self.energy_kwh > self.soc_min_kwh:
            # Сколько полезной энергии мы можем ОТДАТЬ в нагрузку в пределах SOC:
            # внутренний запас = (energy_kwh - soc_min_kwh),
            # но полезная отдаваемая = запас * eta_discharge (теряем часть в преобразованиях).
            discharge_energy_possible = (self.energy_kwh - self.soc_min_kwh) * self.eta_discharge

            # Переводим возможную энергию в допустимую мощность на шаге dt_hours.
            discharge_kw = min(discharge_kw, discharge_energy_possible / max(dt_hours, 1e-9))
        else:
            # Если батарея уже на нижней границе — разряд запрещен.
            discharge_kw = 0.0

        # ----------------- Обновление энергии -----------------
        # (C) Учет КПД: заряд прибавляет, разряд вычитает (с делением на eta_discharge).
        delta_kwh = (
            charge_kw * self.eta_charge * dt_hours
            - discharge_kw / max(self.eta_discharge, 1e-9) * dt_hours
        )

        # Обновляем состояние и прижимаем к границам SOC.
        self.energy_kwh = _round_energy(
            max(self.soc_min_kwh, min(self.soc_max_kwh, self.energy_kwh + delta_kwh))
        )

        return charge_kw, discharge_kw, self.energy_kwh

    def as_dict(self) -> dict:
        """
        Сериализация состояния в dict (удобно для сохранения метаданных, отчетов, логов).

        Важно:
        • ключ 'soc_start_kwh' здесь отражает ТЕКУЩЕЕ значение energy_kwh на момент вызова.
          Несмотря на слово 'start' в имени, это именно "текущий SOC".
          Если хочешь хранить отдельно старт и конец горизонта, это лучше делать снаружи.
        """
        return {
            "capacity_kwh": self.capacity_kwh,
            "soc_min_kwh": self.soc_min_kwh,
            "soc_max_kwh": self.soc_max_kwh,
            "eta_charge": self.eta_charge,
            "eta_discharge": self.eta_discharge,
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "soc_start_kwh": self.energy_kwh,
        }
