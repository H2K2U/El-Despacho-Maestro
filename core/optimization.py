# File: core/optimization.py
"""
Публичный фасад (facade) для запуска оптимизации.

Роль модуля:
1) Дать единый стабильный интерфейс для UI: OptimizationEngine.optimize(...);
2) Спрятать детали: где лежат алгоритмы, как они импортируются;
3) Поддержать два стиля вызова:
   • прямые методы (greedy_optimization / wolf_optimization / de_optimization);
   • выбор по строке через optimize(..., algorithm="...") и AlgorithmFactory.

Важно по матмодели (в терминах твоего проекта):
• upper-level (sizing): выбор типа/количества ВЭУ и АКБ;
• lower-level (dispatch): почасовая диспетчеризация, которую сейчас делает greedy;
• DE/GWO используют greedy как “оценщик” кандидатов и минимизируют штрафную целевую функцию:
  J = 1e9*E_unserved + Fuel + w_dump*Dump + w_hours*HoursDiesel + w_starts*Starts + w_overlap*Overlap + w_capex*Proxy,
  при ограничениях по Pmin/Pmax ДГУ, SOCmin/SOCmax АКБ, P_charge/P_discharge и балансе мощности.
"""

from typing import List

# AlgorithmFactory отвечает за “строка → функция”.
from core.algorithm_factory import AlgorithmFactory

# Импортируем конкретные реализации алгоритмов.
# Это нужно для:
# 1) прямых оберток ниже (OptimizationEngine.greedy_optimization и т.д.);
# 2) fallback-сценариев (если фабрика вернула greedy или нужно прямое обращение).
from core.algorithms import greedy_optimization, wolf_optimization, de_optimization

# Импортируем модели данных (типы) для читабельных сигнатур и подсказок IDE.
from core.models import Battery, DieselPlant, HydroPlant, LoadProfile, OptimizationResult, WindTurbine


class OptimizationEngine:
    """Публичный фасад для запуска оптимизации."""

    @staticmethod
    def greedy_optimization(
        load_profile: LoadProfile,
        diesel_plants: List[DieselPlant],
        hydro_plants: List[HydroPlant],
        wind_turbines: List[WindTurbine],
        batteries: List[Battery],
        wind_speeds=None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Прямой запуск жадной диспетчеризации (нижний уровень).

        Параметры:
        • load_profile — почасовая нагрузка (DataFrame с колонкой "load" и индексом времени);
        • diesel_plants — список ДЭС, каждая содержит набор ДГУ;
        • hydro_plants — список МГЭС (обычно мощность зависит от месяца по гидрографу);
        • wind_turbines — доступные ВЭУ (в greedy обычно берется “выбранная”/первая, см. greedy.py);
        • batteries — доступные АКБ (аналогично);
        • wind_speeds — ряд скоростей ветра (list/Series/или DataFrame в других алгоритмах);
        • kwargs — любые доп. параметры (например, progress_cb, initial_soc_kwh, verbose и т.п.).

        Возвращает:
        • OptimizationResult, где schedule — таблица почасовых мощностей/состояний.
        """
        return greedy_optimization(
            load_profile, diesel_plants, hydro_plants, wind_turbines, batteries, wind_speeds, **kwargs
        )

    @staticmethod
    def wolf_optimization(
        load_profile: LoadProfile,
        diesel_plants: List[DieselPlant],
        hydro_plants: List[HydroPlant],
        wind_turbines: List[WindTurbine],
        batteries: List[Battery],
        wind_speeds=None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Прямой запуск GWO (Grey Wolf Optimizer).

        Смысл:
        • GWO подбирает верхнеуровневые переменные (тип/кол-во ВЭУ и АКБ);
        • оценку кандидата дает greedy_optimization через штрафную целевую функцию;
        • результатом возвращается лучший найденный schedule (обычно после FULL-проверки).
        """
        return wolf_optimization(
            load_profile, diesel_plants, hydro_plants, wind_turbines, batteries, wind_speeds, **kwargs
        )

    @staticmethod
    def de_optimization(
        load_profile: LoadProfile,
        diesel_plants: List[DieselPlant],
        hydro_plants: List[HydroPlant],
        wind_turbines: List[WindTurbine],
        batteries: List[Battery],
        wind_speeds=None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Прямой запуск DE (Differential Evolution).

        Смысл:
        • DE так же решает верхнеуровневую задачу выбора состава (sizing);
        • для каждого кандидата считает greedy-диспетчеризацию и фитнес;
        • может делать FAST/FULL (быстрые окна + периодическая полная проверка).
        """
        return de_optimization(
            load_profile, diesel_plants, hydro_plants, wind_turbines, batteries, wind_speeds, **kwargs
        )

    @staticmethod
    def optimize(
        load_profile: LoadProfile,
        diesel_plants: List[DieselPlant],
        hydro_plants: List[HydroPlant],
        wind_turbines: List[WindTurbine],
        batteries: List[Battery],
        algorithm: str = "greedy",
        **kwargs,
    ) -> OptimizationResult:
        """
        Основной “универсальный” метод оптимизации: выбирает алгоритм по строке.

        Как это работает:
        1) algorithm — строковый ключ ("greedy", "wolf", "de");
        2) AlgorithmFactory.get_algorithm(algorithm) возвращает функцию;
        3) Эта функция вызывается с единым набором аргументов:
           (load_profile, diesel_plants, hydro_plants, wind_turbines, batteries, **kwargs)

        Важно:
        • Все специфические параметры алгоритмов (например, n_wolves, generations,
          wind_speeds / wind_data, progress_cb, seed и т.д.) передаются через kwargs;
        • Если ключ неизвестен — фабрика вернет greedy_optimization (fallback).
        """
        algorithm_callable = AlgorithmFactory.get_algorithm(algorithm)

        # Вызываем выбранный алгоритм единообразно.
        # Это и есть “фасад”: UI не обязан знать сигнатуру каждого метода — он просто шлет kwargs.
        return algorithm_callable(
            load_profile, diesel_plants, hydro_plants, wind_turbines, batteries, **kwargs
        )
