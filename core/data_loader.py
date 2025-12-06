# File: core/data_loader.py
import pandas as pd
import numpy as np
from typing import List

from PySide6.QtWidgets import QFileDialog, QMessageBox, QDialogButtonBox

from .models import LoadProfile, WindTurbine, Battery, DieselUnit, HydroPlant


class DataLoader:
    """
    Класс-утилита для загрузки входных данных проекта из файлов (Excel/CSV).

    Что здесь загружается:
    • Профиль нагрузки (график потребления во времени) → LoadProfile;
    • Каталог ветротурбин (ВЭУ) → список WindTurbine (в т.ч. с кривыми мощности);
    • Данные ветра (скорости ветра по времени) → DataFrame c 'wind_speed';
    • Каталог аккумуляторов (АКБ) → список Battery;
    • Кривая расхода топлива ДЭУ → DataFrame (как есть);
    • Гидрограф (расход/приток по месяцам) → dict {месяц: расход}.

    Важный момент по стилю:
    • Все методы, кроме load_hydro_graph_excel, сделаны staticmethod-ами
      (их можно вызывать без создания объекта DataLoader).

    Потенциальная проблема:
    • Здесь импортированы QFileDialog, QMessageBox, QDialogButtonBox,
      но в этом файле они вообще не используются. Это может быть:
      - "остатки" от GUI-кода;
      - план на будущее.
      Если они не нужны — лучше удалить импорты, чтобы не тянуть UI-зависимости в ядро.
    """

    @staticmethod
    def safe_load_dataframe(file_path: str, required_columns: list) -> pd.DataFrame:
        """
        Безопасная загрузка DataFrame и проверка обязательных колонок.

        Параметры:
        • file_path — путь к файлу (.xlsx или .csv);
        • required_columns — список названий колонок, которые должны присутствовать.

        Логика:
        1) Определяем формат:
           - если путь оканчивается на '.xlsx' → pd.read_excel;
           - иначе → pd.read_csv (предполагается CSV).
        2) Проверяем, что все required_columns есть в df.columns.
           Если каких-то нет — кидаем ValueError с перечислением.
        3) Возвращаем DataFrame.

        Почему это полезно:
        • Любая дальнейшая логика (парсинг времени, конвертация чисел)
          предполагает, что нужные столбцы реально существуют.
        """
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path)

            # missing_columns — список тех обязательных колонок, которых нет в файле.
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

            return df
        except Exception as e:
            # Оборачиваем любую ошибку в ValueError с понятным сообщением:
            # • где проблема (в каком файле),
            # • и оригинальное сообщение исключения.
            raise ValueError(f"Ошибка загрузки файла {file_path}: {str(e)}")

    @staticmethod
    def load_load_profile(file_path: str) -> LoadProfile:
        """
        Загрузка графика нагрузки из Excel или CSV.

        Ожидаемый формат входного файла:
        • Колонка 'Время' — дата/время (строка или уже datetime);
        • Колонка 'Потребление' — мощность нагрузки (кВт), возможно с запятыми.

        Что на выходе:
        • объект LoadProfile(data=df),
          где df — DataFrame с индексом времени и колонкой 'load'.

        Детали преобразования:
        1) Читаем df через safe_load_dataframe(file_path, ['Время','Потребление']).
        2) Парсим df['Время'] в datetime:
           - пробуем формат '%d.%m.%Y %H:%M'
           - если не вышло → '%d.%m.%Y %H:%M:%S'
           - если и это не вышло → pd.to_datetime без явного формата (автоопределение).
        3) Приводим потребление к float:
           - заменяем ',' на '.' (частая проблема при русской локали);
           - .astype(float).
        4) set_index('Время') и переименовываем колонку в 'load'.

        Важно:
        • Вся дальнейшая оптимизация у тебя ожидает именно 'load' (см. greedy/wolf/de).
        """
        try:
            df = DataLoader.safe_load_dataframe(file_path, ['Время', 'Потребление'])

            # Преобразуем время: сначала пробуем строгие форматы, потом "как получится".
            try:
                df['Время'] = pd.to_datetime(df['Время'], format='%d.%m.%Y %H:%M')
            except:
                try:
                    df['Время'] = pd.to_datetime(df['Время'], format='%d.%m.%Y %H:%M:%S')
                except:
                    df['Время'] = pd.to_datetime(df['Время'])

            # 'Потребление' может быть строкой вида "123,45" → заменяем запятую и делаем float.
            df['Потребление'] = df['Потребление'].astype(str).str.replace(',', '.').astype(float)

            # Индекс времени нужен, чтобы:
            # • красиво срезать окна;
            # • корректно вычислять dt;
            # • работать с месяцами (month) в МГЭС.
            df.set_index('Время', inplace=True)

            # Внутренняя конвенция проекта: колонка нагрузки называется 'load'.
            df.rename(columns={'Потребление': 'load'}, inplace=True)

            return LoadProfile(data=df)

        except Exception as e:
            raise ValueError(f"Ошибка загрузки графика нагрузки: {str(e)}")

    @staticmethod
    def load_wind_turbines(file_path: str) -> List[WindTurbine]:
        """
        Загрузка каталога ВЭУ (ветроустановок) в "правильном формате" с кривыми мощности.

        Предполагаемый формат файла (по коду):
        • Строка описывает одну турбину.
        • Базовые поля:
          - 'Turbine'              — имя/модель;
          - 'Rated_Power_kW'       — номинальная мощность (кВт);
          - 'Hub_Height_m'         — высота втулки (м);
          - 'Cut_in_Wind_Speed_m/s' (опционально) — скорость включения, default=3;
          - 'Rated_Wind_Speed_m/s'  (опционально) — номинальная скорость, default=15;
          - 'Cut_out_Wind_Speed_m/s' (опционально) — скорость отключения, default=25.
        • И самое интересное: колонки скорости ветра 0..25 (строки "0","1","2"... "25"),
          в которых лежит мощность при данной скорости.

        Что делает код:
        1) pd.read_excel(file_path) → df
        2) Для каждой строки:
           - собирает список power_curve_data: [{'wind_speed': v, 'power': P(v)}]
           - преобразует в DataFrame power_curve
           - создает WindTurbine с этой power_curve
        3) Возвращает список turbines.

        Риск/нюанс:
        • Проверка "if speed_str in row" — работает, потому что row — Series и "in" проверяет индекс,
          но читается двусмысленно. Чище: `if speed_str in df.columns`.
        • speed_str = str(speed) if speed == int(speed) ... — тут speed всегда int, так что ветвление лишнее.
        """
        try:
            df = pd.read_excel(file_path)
            turbines = []

            for _, row in df.iterrows():
                # power_curve_data — список точек кривой мощности:
                # каждой скорости соответствует мощность (обычно кВт).
                power_curve_data = []

                # Проходим по целочисленным скоростям 0..25 м/с
                for speed in range(0, 26):
                    speed_str = str(speed) if speed == int(speed) else str(speed)

                    # Здесь проверяем: есть ли в строке значение для колонки "speed_str".
                    # (По сути проверяется наличие в заголовках таблицы.)
                    if speed_str in row:
                        power = row[speed_str]
                        if pd.notna(power):
                            power_curve_data.append({
                                'wind_speed': speed,
                                'power': float(power)
                            })

                power_curve = pd.DataFrame(power_curve_data)

                # Создаем объект WindTurbine (модель ВЭУ в твоем проекте).
                turbine = WindTurbine(
                    name=row['Turbine'],
                    nominal_power=row['Rated_Power_kW'],
                    power_curve=power_curve,
                    height=row['Hub_Height_m'],
                    # row.get(...) — безопасно: если столбца нет, подставим дефолт.
                    cut_in_speed=row.get('Cut_in_Wind_Speed_m/s', 3),
                    rated_speed=row.get('Rated_Wind_Speed_m/s', 15),
                    cut_out_speed=row.get('Cut_out_Wind_Speed_m/s', 25)
                )
                turbines.append(turbine)

            return turbines

        except Exception as e:
            raise ValueError(f"Ошибка загрузки каталога ВЭУ: {str(e)}")

    @staticmethod
    def load_wind_data(file_path: str) -> pd.DataFrame:
        """
        Загрузка данных о скорости ветра (временной ряд).

        Ожидаемый формат:
        • 'Время' — дата/время;
        • 'Скорость' — скорость ветра (м/с), возможно как строка с запятой.

        Выход:
        • DataFrame с индексом времени и колонкой 'wind_speed'.

        Зачем так:
        • Алгоритмы (greedy/wolf/de) берут wind_speeds как список или wind_data как DataFrame,
          и внутри вызывают available_power(speed) у WindTurbine.
        """
        try:
            df = DataLoader.safe_load_dataframe(file_path, ['Время', 'Скорость'])

            # Парс времени — та же трехступенчатая схема, как для нагрузки.
            try:
                df['Время'] = pd.to_datetime(df['Время'], format='%d.%m.%Y %H:%M')
            except:
                try:
                    df['Время'] = pd.to_datetime(df['Время'], format='%d.%m.%Y %H:%M:%S')
                except:
                    df['Время'] = pd.to_datetime(df['Время'])

            # Скорость может быть "5,2" → делаем "5.2" → float
            df['Скорость'] = df['Скорость'].astype(str).str.replace(',', '.').astype(float)

            df.set_index('Время', inplace=True)
            df.rename(columns={'Скорость': 'wind_speed'}, inplace=True)

            return df

        except Exception as e:
            raise ValueError(f"Ошибка загрузки данных о ветре: {str(e)}")

    @staticmethod
    def load_batteries(file_path: str) -> List[Battery]:
        """
        Загрузка каталога АКБ из Excel.

        Важная деталь:
        • Читается конкретный лист 'АКБ':
          df = pd.read_excel(file_path, sheet_name='АКБ')

        Ожидаемые колонки (по коду):
        • 'Наименование'
        • 'E_nom_kWh'          — номинальная емкость (кВт*ч)
        • 'P_charge_max_kW'    — максимальная мощность заряда (кВт)
        • 'P_discharge_max_kW' — максимальная мощность разряда (кВт)
        • 'eta_charge'         — КПД (в твоей модели может трактоваться как round-trip или charge-side; см. BatteryState)
        • 'SOC_min'            — минимальный SOC (доля 0..1 или кВт*ч — зависит от модели BatteryState)
        • 'SOC_max'            — максимальный SOC

        Выход:
        • список Battery, где каждую строку таблицы мы превращаем в объект модели.
        """
        try:
            df = pd.read_excel(file_path, sheet_name='АКБ')
            batteries = []

            for _, row in df.iterrows():
                battery = Battery(
                    name=row['Наименование'],
                    capacity=row['E_nom_kWh'],
                    max_charge_power=row['P_charge_max_kW'],
                    max_discharge_power=row['P_discharge_max_kW'],
                    efficiency=row['eta_charge'],
                    soc_min=row['SOC_min'],
                    soc_max=row['SOC_max']
                )
                batteries.append(battery)

            return batteries

        except Exception as e:
            raise ValueError(f"Ошибка загрузки каталога АКБ: {str(e)}")

    @staticmethod
    def load_diesel_curve(file_path: str) -> pd.DataFrame:
        """
        Загрузка кривой расхода топлива ДЭУ.

        Сейчас реализация максимально простая:
        • просто читает Excel и возвращает DataFrame "как есть".

        Нюанс:
        • В отличие от других загрузчиков, тут нет проверки обязательных колонок.
          Это нормально, если дальше в коде есть отдельная валидация,
          иначе при неправильном формате ошибка всплывет позже и будет менее понятной.
        """
        df = pd.read_excel(file_path)
        return df

    def load_hydro_graph_excel(file_path):
        """
        Универсальная загрузка гидрографа из Excel файла.

        Что такое "гидрограф" здесь:
        • словарь hydro_graph: Dict[int, float], где ключ — месяц 1..12,
          а значение — среднемесячный расход/приток/доступная вода (в тех единицах,
          которые понимает твоя HydroPlant.available_power(month)).

        Ожидаемые форматы файла (поддерживаются несколько вариантов):

        Вариант 1 (основной, "две строки"):
        • Первая строка: месяцы (1..12)
        • Вторая строка: расходы/притоки (12 чисел)

        Вариант 2 ("два столбца"):
        • Два столбца: в одном месяцы 1..12, в другом расходы.
          Но тут код работает через транспонирование и поиск пар колонок,
          потому что вход может быть "перевернут".

        Вариант 3 (fallback):
        • если не распарсили структуру — собираем все числа из файла подряд
          и берем первые 12 как значения по месяцам 1..12.

        Важно:
        • Метод НЕ объявлен как @staticmethod, хотя использует только file_path.
          Его можно спокойно сделать staticmethod-ом для единообразия.
        """
        try:
            # Читаем Excel БЕЗ заголовков, потому что формат может быть нестандартный.
            df = pd.read_excel(file_path, header=None)

            hydro_graph = {}

            # ---------- Вариант 1: две строки ----------
            # df.iloc[0] — предполагаемая строка месяцев
            # df.iloc[1] — предполагаемая строка расходов
            if len(df) >= 2:
                months_row = df.iloc[0]
                flows_row = df.iloc[1]

                # month_values — найденные числа 1..12 в первой строке
                month_values = []
                for val in months_row:
                    try:
                        month = int(float(val))
                        if 1 <= month <= 12:
                            month_values.append(month)
                    except:
                        # если значение не число — пропускаем
                        continue

                # Если похоже на месяцы (>=10 из 12 — допускаем неполное заполнение),
                # то считаем, что это нужный формат.
                if len(month_values) >= 10:
                    for i, month in enumerate(month_values):
                        if i < len(flows_row):
                            try:
                                flow = float(flows_row.iloc[i])
                                hydro_graph[month] = flow
                            except:
                                hydro_graph[month] = 0
                else:
                    # ---------- Вариант 2: два столбца (после транспонирования) ----------
                    df_transposed = df.T

                    # Ищем пару колонок (col1=месяцы, col2=расходы)
                    for i in range(min(10, len(df_transposed.columns))):
                        col1 = df_transposed.iloc[:, i] if i < len(df_transposed.columns) else None
                        col2 = df_transposed.iloc[:, i + 1] if i + 1 < len(df_transposed.columns) else None

                        if col1 is not None and col2 is not None:
                            month_values = []
                            for val in col1:
                                try:
                                    month = int(float(val))
                                    if 1 <= month <= 12:
                                        month_values.append(month)
                                except:
                                    continue

                            if len(month_values) >= 10:
                                for j, month in enumerate(month_values):
                                    if j < len(col2):
                                        try:
                                            flow = float(col2.iloc[j])
                                            hydro_graph[month] = flow
                                        except:
                                            hydro_graph[month] = 0
                                break

            # ---------- Вариант 3: fallback "первые 12 чисел" ----------
            if not hydro_graph:
                all_values = []
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        try:
                            all_values.append(float(df.iloc[i, j]))
                        except:
                            pass

                if len(all_values) >= 12:
                    for month in range(1, 13):
                        if month - 1 < len(all_values):
                            hydro_graph[month] = all_values[month - 1]

            # ---------- Заполняем пропущенные месяцы нулями ----------
            # Это важно, чтобы downstream-код не падал на month not in dict.
            for month in range(1, 13):
                if month not in hydro_graph:
                    hydro_graph[month] = 0.0

            return hydro_graph

        except Exception as e:
            raise ValueError(f"Ошибка загрузки гидрографа: {str(e)}")
