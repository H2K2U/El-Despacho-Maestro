# File: ui/main_window.py
"""
Главное окно приложения (MainWindow) — "скелет" UI и место, где собираются все виджеты.

Этот модуль НИЧЕГО "не оптимизирует" сам по себе:
• он предоставляет пользователю интерфейс для:
  — загрузки входных данных (нагрузка, ветер);
  — управления каталогом оборудования (ДЭУ/ДЭС/МГЭС/ВЭУ/АКБ);
  — выбора алгоритма оптимизации и запуска расчета;
  — выбора даты/дня для визуализации;
  — просмотра графиков и таблиц результата (EnergyChart/BatteryChart, таблицы итогов).

Фактически MainWindow — это:
1) раскладка (layouts/splitter/tabs);
2) создание и хранение ссылок на controls (кнопки/списки/таблицы);
3) "тонкие" методы обновления UI (update_*), которые принимают уже готовые данные.

Важно:
• В этом файле почти нет сигналов/слотов (кроме прогресса), то есть
  он сейчас больше "view" чем "controller".
  Ожидается, что снаружи (например, в run.py или controller) будут сделаны connect():
  — load_load_profile_btn.clicked -> загрузчик данных;
  — optimize_btn.clicked -> запуск оптимизации;
  — date_selector.dateChanged -> перерисовка графика по выбранной дате;
  — add_*_btn -> открытие диалогов создания оборудования;
  и т.д.

---

Зависимости:
• PySide6 — UI;
• matplotlib backend для Qt — косвенно, через core.visualization.EnergyChart/BatteryChart;
• faulthandler включен в начале — чтобы при крашах Qt/Python увидеть стек-трейс в консоли.

---

Структура UI:
• QSplitter (горизонтальный):
  — слева: "панель управления" (данные, оборудование, оптимизация, выбор даты);
  — справа: вкладки с графиками/таблицами результата.
• Tabs справа:
  1) "График покрытия нагрузки" — EnergyChart внутри ScrollArea;
  2) "АКБ" — BatteryChart + таблица параметров АКБ;
  3) "Подобранное оборудование" — таблица выбранного состава;
  4) "Результаты оптимизации" — агрегированные метрики;
  5) "Экономический анализ" — пока заглушка.

---

Очень важная деталь:
• Прогресс-бар обновляется асинхронно через QTimer (каждые 75 мс).
  Это сделано для того, чтобы "тяжелый" расчет мог дергать callback _on_progress(),
  а UI не залипал и не получал спам setValue() слишком часто.

"""

import faulthandler
faulthandler.enable()

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QComboBox,
    QTableWidget, QTableWidgetItem,
    QGroupBox, QFormLayout, QSplitter, QProgressBar,
    QScrollArea, QSizePolicy,
    QListWidget, QListWidgetItem, QHeaderView,
    QDateEdit
)
from PySide6.QtCore import Qt, QTimer, QDate
from PySide6.QtGui import QFont

from core.visualization import EnergyChart, BatteryChart


class MainWindow(QMainWindow):
    """
    Главное окно приложения.

    Поля экземпляра (ключевые):
    • self.splitter — делит окно на левую/правую часть;
    • self.left_panel / self.right_panel — контейнеры;
    • self.tabs — вкладки результатов;
    • self.visualization_widget — EnergyChart (покрытие нагрузки);
    • self.battery_chart — BatteryChart (SOC и Pcharge/Pdischarge);
    • self.progress_bar — для отображения прогресса оптимизации (скрыт по умолчанию);
    • self.date_selector — выбор даты, на основе которой выбирается "день" из расписания.

    Потенциальная интеграция:
    — контроллер может получить доступ к кнопкам/спискам/таблицам через эти поля
      и навесить обработчики.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("El Despacho Maestro")

        # -------------------------
        # 1) Геометрия окна и базовая конфигурация
        # -------------------------
        # Начальный размер окна. Пользователь все равно сможет изменить.
        self.resize(1400, 800)

        # Минимальный размер: защищает от "схлопывания" UI, когда элементы перестают помещаться.
        self.setMinimumSize(1024, 600)

        # -------------------------
        # 2) Центральный виджет и корневой layout
        # -------------------------
        # В Qt QMainWindow требует setCentralWidget() — туда кладем корневой контейнер.
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Главный layout окна: горизонтальный, потому что внутри будет QSplitter.
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(10)

        # -------------------------
        # 3) Сплиттер: левая панель (управление) + правая панель (результаты)
        # -------------------------
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        # Левая панель — узкая, с ограничением: не даем ей разрастаться.
        self.left_panel = QWidget()
        self.left_panel.setMinimumWidth(350)
        self.left_panel.setMaximumWidth(500)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        self.splitter.addWidget(self.left_panel)

        # Правая панель — основное пространство для графиков/таблиц.
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(5, 5, 5, 5)
        self.splitter.addWidget(self.right_panel)

        # Начальное соотношение размеров: примерно 30% слева и 70% справа.
        # (значения в пикселях, но Qt интерпретирует их как "веса" при начальном размещении)
        self.splitter.setSizes([420, 980])

        # Сборка внутренних панелей
        self.setup_left_panel(self.left_layout)
        self.setup_right_panel(self.right_layout)

        # Флаг "левая панель показана". Сейчас не используется, но задел для toggle.
        self.left_panel_visible = True

        # -------------------------
        # 4) Механизм прогресса: буферизация обновлений через таймер
        # -------------------------
        # Состояние прогресса хранится как кортеж:
        # (done, total, msg)
        # done/total — целые числа; msg — сообщение (например "day 12/365").
        self._progress_state = (0, 1, "")
        self._progress_active = False

        # Таймер каждые 75 мс синхронизирует self._progress_state -> progress_bar.
        # Это снижает дергание UI при частых апдейтах прогресса.
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(75)
        self._progress_timer.timeout.connect(self._flush_progress_to_ui)
        self._progress_timer.start()

    # -------------------------------------------------------------------------
    # ПРОГРЕСС
    # -------------------------------------------------------------------------
    def _on_progress(self, done: int, total: int, msg: str = ""):
        """
        Callback для расчета/оптимизации.

        Как использовать снаружи:
        — оптимизатор/диспетчеризация может принимать progress_cb и дергать:
            progress_cb(done=i, total=N, msg="...")

        Что делает метод:
        — не трогает UI напрямую, а кладет значения в буфер self._progress_state.
          UI обновится в _flush_progress_to_ui() по таймеру.
        """
        self._progress_state = (done, total, msg or "")
        self._progress_active = True

    def _flush_progress_to_ui(self):
        """
        Синхронизация буфера прогресса с реальным виджетом QProgressBar.

        Логика:
        1) вычисляем процент;
        2) если прогресс активен — показываем progress_bar и обновляем текст;
        3) если done >= total — считаем прогресс завершенным и перестаем обновлять,
           но progress_bar при этом остается видимой, пока кто-то снаружи не скроет ее
           или пока новый прогресс не начнется.

        Потенциальное улучшение:
        — после done>=total можно скрыть progress_bar через QTimer.singleShot(500, hide),
          чтобы UI выглядел аккуратнее.
        """
        done, total, msg = self._progress_state
        pct = int(100 * done / max(1, total))

        if self._progress_active:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
            self.progress_bar.setFormat(f"{pct}% — {msg}" if msg else f"{pct}%")

            if done >= total:
                self._progress_active = False

    # -------------------------------------------------------------------------
    # ЛЕВАЯ ПАНЕЛЬ (управление)
    # -------------------------------------------------------------------------
    def setup_left_panel(self, layout):
        """
        Сборка левой панели.

        UI блоки:
        1) "Загрузка данных":
           — кнопки загрузки профиля нагрузки и данных ветра;
           — статусы (Label) для пользователя.
        2) "Оборудование":
           — кнопки добавления/загрузки разных сущностей;
           — списки (QListWidget) для визуального контроля, что уже добавлено.
        3) "Оптимизация":
           — выбор алгоритма;
           — кнопка "Запустить оптимизацию";
           — progress_bar.
        4) "Визуализация":
           — выбор даты (QDateEdit). Сейчас жестко ограничено 2021 годом.

        Тут нет обработчиков событий: они должны быть подключены в контроллере.
        """
        # -------------------------
        # 1) Группа загрузки данных
        # -------------------------
        data_group = QGroupBox("Загрузка данных")
        data_layout = QVBoxLayout(data_group)

        # Кнопка загрузки графика нагрузки.
        self.load_load_profile_btn = QPushButton("Загрузить график нагрузки")

        # Статус: нужен, чтобы пользователь понимал, загружены ли данные.
        self.load_profile_status = QLabel("Не загружен")

        # Кнопка загрузки ветровых данных.
        self.load_wind_data_btn = QPushButton("Загрузить данные о ветре")

        # Статус ветра отдельно.
        self.wind_data_status = QLabel("Не загружены")

        data_layout.addWidget(self.load_load_profile_btn)
        data_layout.addWidget(self.load_profile_status)
        data_layout.addWidget(self.load_wind_data_btn)
        data_layout.addWidget(self.wind_data_status)

        # -------------------------
        # 2) Группа оборудования
        # -------------------------
        equipment_group = QGroupBox("Оборудование")
        equipment_layout = QVBoxLayout(equipment_group)

        # Кнопки изменения "каталога/набора" оборудования.
        # Логика обычно такая:
        # — add_diesel_unit_btn -> открыть DieselUnitDialog и добавить в список diesel_units;
        # — create_diesel_plant_btn -> открыть DieselPlantDialog (выбрать из дизелей) и добавить diesel_plants;
        # — add_hydro_plant_btn -> открыть HydroPlantDialog и добавить в hydro_plants;
        # — load_wind_turbines_btn -> загрузить каталог ВЭУ из файла (Excel);
        # — load_batteries_btn -> загрузить каталог АКБ из файла (Excel).
        self.add_diesel_unit_btn = QPushButton("Добавить ДЭУ")
        self.create_diesel_plant_btn = QPushButton("Создать ДЭС")
        self.add_hydro_plant_btn = QPushButton("Добавить МГЭС")
        self.load_wind_turbines_btn = QPushButton("Загрузить каталог ВЭУ")
        self.load_batteries_btn = QPushButton("Загрузить каталог АКБ")

        # Списки отображают текущие элементы.
        # Они не содержат объектов (Qt.UserRole) — только текст.
        # Если понадобится редактирование/удаление — лучше хранить объект в item.setData.
        self.diesel_units_list = QListWidget()
        self.diesel_plants_list = QListWidget()
        self.hydro_plants_list = QListWidget()
        self.wind_turbines_list = QListWidget()
        self.batteries_list = QListWidget()

        equipment_layout.addWidget(QLabel("ДЭУ:"))
        equipment_layout.addWidget(self.diesel_units_list)
        equipment_layout.addWidget(self.add_diesel_unit_btn)

        equipment_layout.addWidget(QLabel("ДЭС:"))
        equipment_layout.addWidget(self.diesel_plants_list)
        equipment_layout.addWidget(self.create_diesel_plant_btn)

        equipment_layout.addWidget(QLabel("МГЭС:"))
        equipment_layout.addWidget(self.hydro_plants_list)
        equipment_layout.addWidget(self.add_hydro_plant_btn)

        equipment_layout.addWidget(QLabel("ВЭУ:"))
        equipment_layout.addWidget(self.wind_turbines_list)
        equipment_layout.addWidget(self.load_wind_turbines_btn)

        equipment_layout.addWidget(QLabel("АКБ:"))
        equipment_layout.addWidget(self.batteries_list)
        equipment_layout.addWidget(self.load_batteries_btn)

        # -------------------------
        # 3) Группа оптимизации
        # -------------------------
        optimization_group = QGroupBox("Оптимизация")
        optimization_layout = QFormLayout(optimization_group)

        # Выпадающий список: хранит "человекочитаемое" название + userData с кодом алгоритма.
        # userData используется, чтобы потом вызвать OptimizationEngine.optimize(..., algorithm=code)
        self.algorithm_selector = QComboBox()
        self.algorithm_selector.addItem("Жадный алгоритм", userData="greedy")
        self.algorithm_selector.addItem("Метод серых волков", userData="wolf")
        self.algorithm_selector.addItem("Дифференциальная эволюция (DE)", userData="de")

        # Кнопка старта.
        self.optimize_btn = QPushButton("Запустить оптимизацию")
        self.optimize_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }"
        )

        # ProgressBar скрыт пока нет активного расчета.
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        optimization_layout.addRow("Алгоритм:", self.algorithm_selector)
        optimization_layout.addRow(self.optimize_btn)
        optimization_layout.addRow(self.progress_bar)

        # -------------------------
        # 4) Группа выбора дня/даты визуализации
        # -------------------------
        day_group = QGroupBox("Визуализация")
        day_layout = QFormLayout(day_group)

        # date_selector управляет выбором "дня" для показа.
        # В текущем виде диапазон жестко зафиксирован на 2021.
        # Это работает, если входные данные тоже "2021", но ломается для других годов.
        # Практическое улучшение:
        # — после загрузки load_profile (Index) выставлять min/max по реальным датам.
        self.date_selector = QDateEdit()
        self.date_selector.setDisplayFormat("dd.MM.yyyy")
        self.date_selector.setCalendarPopup(True)
        self.date_selector.setMinimumDate(QDate(2021, 1, 1))
        self.date_selector.setMaximumDate(QDate(2021, 12, 31))
        self.date_selector.setDate(QDate(2021, 1, 1))
        day_layout.addRow("Дата:", self.date_selector)

        # Добавляем группы на левую панель сверху вниз.
        layout.addWidget(data_group)
        layout.addWidget(equipment_group)
        layout.addWidget(optimization_group)
        layout.addWidget(day_group)

        # Stretch внизу "прижимает" группы к верху и делает свободное пространство снизу.
        layout.addStretch()

    # -------------------------------------------------------------------------
    # ПРАВАЯ ПАНЕЛЬ (результаты)
    # -------------------------------------------------------------------------
    def setup_right_panel(self, layout):
        """
        Сборка правой панели: вкладки результатов.

        Вкладки:
        1) График покрытия нагрузки (EnergyChart);
        2) АКБ (BatteryChart + таблица);
        3) Подобранное оборудование (таблица);
        4) Результаты оптимизации (таблица);
        5) Экономический анализ (таблица-заглушка).
        """
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        layout.addWidget(self.tabs)

        # -------------------------
        # TAB 1: Покрытие нагрузки
        # -------------------------
        self.chart_tab = QWidget()
        chart_layout = QVBoxLayout(self.chart_tab)
        chart_layout.setContentsMargins(5, 5, 5, 5)

        # Заголовок над графиком (текущая дата/описание)
        self.chart_header = QLabel()
        self.chart_header.setFont(QFont("Arial", 12, QFont.Bold))
        chart_layout.addWidget(self.chart_header)

        # ScrollArea — чтобы график можно было прокручивать, если виджет большой.
        chart_scroll = QScrollArea()
        chart_scroll.setWidgetResizable(True)
        chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        chart_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # EnergyChart — matplolib canvas с отрисовкой компонент (ГЭС/ВЭУ/ДЭС/АКБ/сброс + линия нагрузки).
        self.visualization_widget = EnergyChart(self.chart_tab)
        chart_scroll.setWidget(self.visualization_widget)

        chart_layout.addWidget(chart_scroll)
        self.tabs.addTab(self.chart_tab, "График покрытия нагрузки")

        # -------------------------
        # TAB 2: АКБ
        # -------------------------
        self.battery_tab = QWidget()
        battery_layout = QVBoxLayout(self.battery_tab)
        battery_layout.setContentsMargins(5, 5, 5, 5)

        self.battery_header = QLabel()
        self.battery_header.setFont(QFont("Arial", 12, QFont.Bold))
        battery_layout.addWidget(self.battery_header)

        # Контент: слева график, справа таблица параметров.
        battery_content = QHBoxLayout()
        battery_layout.addLayout(battery_content)

        battery_chart_scroll = QScrollArea()
        battery_chart_scroll.setWidgetResizable(True)
        battery_chart_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        battery_chart_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.battery_chart = BatteryChart(self.battery_tab)
        battery_chart_scroll.setWidget(self.battery_chart)
        battery_content.addWidget(battery_chart_scroll, stretch=3)

        # Таблица сведений по профилю АКБ (soc_start/end, charge/discharge, warnings, источник soc_start).
        self.battery_table = QTableWidget()
        self.battery_table.setColumnCount(2)
        self.battery_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.battery_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.battery_table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        battery_content.addWidget(self.battery_table, stretch=1)

        self.tabs.addTab(self.battery_tab, "АКБ")

        # -------------------------
        # TAB 3: Подобранное оборудование
        # -------------------------
        self.equipment_tab = QWidget()
        equipment_tab_layout = QVBoxLayout(self.equipment_tab)

        # Таблица выбранного состава (тип/имя/мощность/параметры).
        self.equipment_table = QTableWidget()
        self.equipment_table.setColumnCount(4)
        self.equipment_table.setHorizontalHeaderLabels(["Тип", "Наименование", "Мощность/Емкость", "Параметры"])
        self.equipment_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        equipment_tab_layout.addWidget(self.equipment_table)

        self.tabs.addTab(self.equipment_tab, "Подобранное оборудование")

        # -------------------------
        # TAB 4: Результаты оптимизации
        # -------------------------
        self.results_tab = QWidget()
        results_layout = QVBoxLayout(self.results_tab)

        # Таблица агрегированных показателей (топливо, доля ВИЭ, выработки и т.д.).
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(["Параметр", "Значение", "Единицы"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.results_table)

        self.tabs.addTab(self.results_tab, "Результаты оптимизации")

        # -------------------------
        # TAB 5: Экономика (заглушка)
        # -------------------------
        self.economics_tab = QWidget()
        economics_layout = QVBoxLayout(self.economics_tab)

        self.economics_table = QTableWidget()
        self.economics_table.setColumnCount(3)
        self.economics_table.setHorizontalHeaderLabels(["Статья затрат", "Сумма", "Примечание"])
        self.economics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        economics_layout.addWidget(self.economics_table)

        self.tabs.addTab(self.economics_tab, "Экономический анализ")

    # -------------------------------------------------------------------------
    # ОБНОВЛЕНИЕ СПИСКОВ ОБОРУДОВАНИЯ (левая панель)
    # -------------------------------------------------------------------------
    def update_equipment_lists(self, diesel_units, diesel_plants, hydro_plants, wind_turbines, batteries):
        """
        Перерисовать списки оборудования на левой панели.

        Вход:
        — diesel_units: List[DieselUnit]
        — diesel_plants: List[DieselPlant]
        — hydro_plants: List[HydroPlant]
        — wind_turbines: List[WindTurbine]
        — batteries: List[Battery]

        Реализация:
        — полностью очищает каждый QListWidget и заполняет заново.
          Это проще всего и надежно, но при больших списках можно оптимизировать диффом.
        """
        self.diesel_units_list.clear()
        for unit in diesel_units:
            self.diesel_units_list.addItem(QListWidgetItem(f"{unit.name} ({unit.nominal_power} кВт)"))

        self.diesel_plants_list.clear()
        for plant in diesel_plants:
            self.diesel_plants_list.addItem(QListWidgetItem(f"{plant.name} ({plant.total_capacity} кВт)"))

        self.hydro_plants_list.clear()
        for plant in hydro_plants:
            self.hydro_plants_list.addItem(QListWidgetItem(f"{plant.name} ({plant.nominal_power} кВт)"))

        self.wind_turbines_list.clear()
        for turbine in wind_turbines:
            self.wind_turbines_list.addItem(QListWidgetItem(f"{turbine.name} ({turbine.nominal_power} кВт)"))

        self.batteries_list.clear()
        for battery in batteries:
            self.batteries_list.addItem(QListWidgetItem(f"{battery.name} ({battery.capacity} кВт*ч)"))

    # -------------------------------------------------------------------------
    # ОБНОВЛЕНИЕ ТАБЛИЦ РЕЗУЛЬТАТОВ (правая панель)
    # -------------------------------------------------------------------------
    def update_results_tables(self, optimization_result):
        """
        Заполнить таблицы:
        • equipment_table (подобранное оборудование);
        • results_table (общие метрики);
        • economics_table (пока заглушка).

        ВНИМАНИЕ: метод ожидает, что optimization_result.selected_equipment — это dict
        с вложенными dict-ами/списками dict-ов (а не реальные объекты моделей).

        Это соответствует твоей текущей архитектуре оптимизации:
        — алгоритмы возвращают OptimizationResult, где selected_equipment сериализован.
        """

        # -------------------------
        # 1) Таблица выбранного оборудования (equipment_table)
        # -------------------------
        selected_eq = optimization_result.selected_equipment
        equipment_data = []

        # ВЭУ: список словарей с ключами name/nominal_power/height...
        for turbine in selected_eq.get('wind_turbines', []):
            equipment_data.append([
                "ВЭУ",
                turbine.get('name', ''),
                f"{turbine.get('nominal_power', 0)} кВт",
                f"Высота: {turbine.get('height', 0)} м"
            ])

        # АКБ: name/capacity/max_charge_power/max_discharge_power...
        for battery in selected_eq.get('batteries', []):
            equipment_data.append([
                "АКБ",
                battery.get('name', ''),
                f"{battery.get('capacity', 0)} кВт*ч",
                f"Заряд: {battery.get('max_charge_power', 0)} кВт, Разряд: {battery.get('max_discharge_power', 0)} кВт"
            ])

        # МГЭС: name/nominal_power/efficiency...
        for hydro in selected_eq.get('hydro_plants', []):
            equipment_data.append([
                "МГЭС",
                hydro.get('name', ''),
                f"{hydro.get('nominal_power', 0)} кВт",
                f"КПД: {hydro.get('efficiency', 0)}"
            ])

        # ДЭС: name/total_capacity/num_units...
        for diesel_plant in selected_eq.get('diesel_plants', []):
            equipment_data.append([
                "ДЭС",
                diesel_plant.get('name', ''),
                f"{diesel_plant.get('total_capacity', 0)} кВт",
                f"Состав: {diesel_plant.get('num_units', 0)} ДЭУ"
            ])

        # Заполнение QTableWidget
        self.equipment_table.setRowCount(len(equipment_data))
        for row, data in enumerate(equipment_data):
            for col, value in enumerate(data):
                self.equipment_table.setItem(row, col, QTableWidgetItem(str(value)))

        # -------------------------
        # 2) Таблица результатов оптимизации (results_table)
        # -------------------------
        schedule = optimization_result.schedule

        # ВНИМАНИЕ: тут есть потенциальный баг согласованности колонок:
        # — В EnergyChart ты читаешь discharge через compute_charge_discharge(), т.е. колонка может быть
        #   'battery_discharge_to_load' или 'discharge_power' и т.п.
        # — Здесь ты жестко суммируешь schedule['battery_discharge'].
        #   Если колонки 'battery_discharge' нет (а есть 'battery_discharge_to_load'), будет KeyError.
        #
        # Практическое исправление:
        # — использовать compute_charge_discharge(schedule) и суммировать discharge_kw * dt_hours.
        # — или хотя бы брать безопасно через schedule.get('battery_discharge', 0).
        results_data = [
            ["Расход топлива", f"{optimization_result.total_fuel_consumption:,.0f}", "л"],
            ["Доля ВИЭ", f"{optimization_result.renewable_energy_ratio * 100:.1f}", "%"],
            ["Общая выработка", f"{schedule['load'].sum():,.0f}", "кВт*ч"],
            ["Выработка ВИЭ", f"{schedule['renewable_to_load'].sum():,.0f}", "кВт*ч"],
            ["Выработка ДЭС", f"{schedule['diesel'].sum():,.0f}", "кВт*ч"],
            ["Энергия через АКБ", f"{schedule['battery_discharge'].sum():,.0f}", "кВт*ч"],
            ["Сброс энергии", f"{schedule['dump'].sum():,.0f}", "кВт*ч"]
        ]

        self.results_table.setRowCount(len(results_data))
        for row, data in enumerate(results_data):
            for col, value in enumerate(data):
                self.results_table.setItem(row, col, QTableWidgetItem(str(value)))

        # -------------------------
        # 3) Экономика (пока константная заглушка)
        # -------------------------
        economics_data = [
            ["Капитальные затраты", "Расчетная", "Зависит от оборудования"],
            ["Эксплуатационные затраты", "Расчетная", "Топливо, обслуживание"],
            ["Срок окупаемости", "Расчетный", "лет"],
            ["Себестоимость электроэнергии", "Расчетная", "руб/кВт*ч"]
        ]

        self.economics_table.setRowCount(len(economics_data))
        for row, data in enumerate(economics_data):
            for col, value in enumerate(data):
                self.economics_table.setItem(row, col, QTableWidgetItem(str(value)))

    # -------------------------------------------------------------------------
    # ФОРМАТИРОВАНИЕ ДАТЫ ДЛЯ ЗАГОЛОВКОВ
    # -------------------------------------------------------------------------
    @staticmethod
    def format_ru_date(date_obj):
        """
        Преобразовать QDate / date-like объект в строку с русским названием месяца.

        Пример:
        — 2021-01-01 -> "1 января 2021 года"

        Замечание:
        — ты используешь "е" вместо "ё" (февраля/декабря и т.д. — тут нет 'ё', все ок).
        — если date_obj будет QDate, у него свойства day()/month()/year(), но ты вызываешь day/month/year
          как атрибуты. В текущем виде это подойдет для datetime.date, но не для QDate.
          Практическое исправление:
          — либо принимать datetime.date;
          — либо для QDate делать: date_obj.day(), date_obj.month(), date_obj.year().
        """
        months = {
            1: "января", 2: "февраля", 3: "марта", 4: "апреля",
            5: "мая", 6: "июня", 7: "июля", 8: "августа",
            9: "сентября", 10: "октября", 11: "ноября", 12: "декабря"
        }
        return f"{date_obj.day} {months.get(date_obj.month, '')} {date_obj.year} года"

    # -------------------------------------------------------------------------
    # ЗАГОЛОВКИ ВКЛАДОК (красивое оформление через QSS)
    # -------------------------------------------------------------------------
    def update_chart_header(self, formatted_date: str):
        """
        Обновить заголовок вкладки графика покрытия нагрузки.

        formatted_date должен быть уже готовой строкой (например из format_ru_date()).
        """
        self.chart_header.setText(f"График покрытия нагрузки - {formatted_date}")
        self.chart_header.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin: 5px;
            }
        """)

    def update_battery_header(self, formatted_date: str):
        """
        Обновить заголовок вкладки батареи.
        """
        self.battery_header.setText(f"Профиль АКБ - {formatted_date}")
        self.battery_header.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin: 5px;
            }
        """)

    # -------------------------------------------------------------------------
    # ТАБЛИЦА СВОДКИ ПО АКБ (правый TAB "АКБ")
    # -------------------------------------------------------------------------
    def update_battery_table(self, summary: dict):
        """
        Заполнить правую таблицу параметров АКБ.

        Вход summary — словарь, который возвращает BatteryChart.plot_battery_day():
        ключи (типично):
        • soc_start, soc_end — кВт*ч;
        • soc_min_kwh, soc_max_kwh — границы (если известны);
        • total_charge_kwh, total_discharge_kwh, delta_soc_kwh — энергетический баланс;
        • eta_used — использованный КПД (если отличается от 1);
        • warnings — список предупреждений (energy_balance mismatch, SOC violation и т.п.);
        • soc_start_source — "provided"/"fallback"/и др.

        Преобразование:
        — собираем список строк rows и затем пишем в QTableWidget.
        """
        rows = []
        rows.append(("soc_start (кВт*ч)", f"{summary.get('soc_start', 0):.2f}"))
        rows.append(("soc_end (кВт*ч)", f"{summary.get('soc_end', 0):.2f}"))

        if summary.get('soc_min_kwh') is not None:
            rows.append(("soc_min (кВт*ч)", f"{summary['soc_min_kwh']:.2f}"))
        if summary.get('soc_max_kwh') is not None:
            rows.append(("soc_max (кВт*ч)", f"{summary['soc_max_kwh']:.2f}"))

        rows.append(("total_charge_kwh", f"{summary.get('total_charge_kwh', 0):.2f}"))
        rows.append(("total_discharge_kwh", f"{summary.get('total_discharge_kwh', 0):.2f}"))
        rows.append(("delta_soc_kwh", f"{summary.get('delta_soc_kwh', 0):+.2f}"))

        if summary.get('eta_used') is not None:
            rows.append(("efficiency_used", f"{summary['eta_used']:.3f}"))

        warnings = summary.get('warnings', [])
        if warnings:
            rows.append(("warnings", ", ".join(warnings)))

        if summary.get('soc_start_source'):
            rows.append(("soc_start_source", summary['soc_start_source']))

        self.battery_table.setRowCount(len(rows))
        for row_idx, (param, value) in enumerate(rows):
            self.battery_table.setItem(row_idx, 0, QTableWidgetItem(str(param)))
            self.battery_table.setItem(row_idx, 1, QTableWidgetItem(str(value)))

    # -------------------------------------------------------------------------
    # ДИАПАЗОН ДАТ (чтобы подстроиться под реальные входные данные)
    # -------------------------------------------------------------------------
    def set_date_range(self, min_date: QDate, max_date: QDate):
        """
        Задать минимальную/максимальную дату для date_selector.

        Зачем это нужно:
        — после загрузки load_profile ты можешь вычислить по индексу min/max даты
          и выставить реальный диапазон, чтобы пользователь не выбирал "пустые" дни.
        """
        self.date_selector.setMinimumDate(min_date)
        self.date_selector.setMaximumDate(max_date)
        self.date_selector.setDate(min_date)

    def get_selected_date(self) -> QDate:
        """
        Получить выбранную пользователем дату (QDate).

        Обычно дальше делается маппинг:
        — QDate -> day_of_year (1..365) -> start_idx = (day-1)*24
        или, если расписание имеет DatetimeIndex:
        — берем срез schedule.loc["2021-01-01":"2021-01-01 23:00"].
        """
        return self.date_selector.date()
