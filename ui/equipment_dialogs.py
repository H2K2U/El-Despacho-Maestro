# File: ui/equipment_dialogs.py
"""
UI-диалоги для создания/редактирования оборудования (PySide6).

Этот модуль решает две задачи:
1) "Сбор исходных данных" от пользователя через формы/таблицы;
2) Преобразование введенных данных в доменные модели из core.models:
   — DieselUnit (ДЭУ: мощность + КПД + кривая расхода);
   — DieselPlant (ДЭС: набор ДЭУ);
   — HydroPlant (МГЭС: мощность + КПД + напор + гидрограф).

Важно: GUI — это "ввод/валидация/сериализация", а НЕ расчеты.
Расчеты (мощность МГЭС по формуле, топливо ДЭС, диспетчеризация) должны жить в core/*,
а UI лишь создает корректные объекты.

---

Структура файла:
• DieselUnitDialog — создание одной ДЭУ, включая редактирование кривой расхода топлива;
• DieselPlantDialog — создание ДЭС путем выбора нескольких ДЭУ из списка;
• HydroPlantDialog — создание МГЭС + обязательная загрузка гидрографа из Excel.

---

Соглашения по единицам:
• DieselUnit.nominal_power — кВт;
• DieselUnit.efficiency — сейчас в модели трактуется как КПД (0..1), но UI подписывает "%".
  Это потенциальная путаница (см. комментарии ниже).
• fuel_curve:
  — в модели ожидается DataFrame с колонками ['load', 'consumption'],
    где load — кВт (абсолютная нагрузка), consumption — л/ч.
  — в таблице UI пользователь вводит "Нагрузка, %" и "Расход, л/ч" (то есть расход уже абсолютный).
  — при get_diesel_unit() нагрузка % конвертируется в кВт: load = load_pct/100 * nominal_power.

• HydroPlant.hydro_graph:
  — словарь {месяц (1..12): расход Q (м^3/с)}.
  — HydroPlant.available_power() потом пересчитает это в кВт через:
    P = ρ * g * H * Q * η   (и ограничение nominal_power).
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QDoubleSpinBox, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QListWidget,
    QListWidgetItem, QGroupBox, QLabel, QDialogButtonBox
)
from PySide6.QtCore import Qt
import pandas as pd

from core.models import DieselUnit, DieselPlant, HydroPlant


class DieselUnitDialog(QDialog):
    """
    Диалог создания одной дизельной установки (ДЭУ).

    UI-логика:
    — имя, номинальная мощность, КПД;
    — таблица кривой расхода топлива (нагрузка %, расход л/ч);
    — кнопки:
      • "Загрузить из файла" — подставить кривую из CSV/XLSX;
      • "Сохранить в файл" — выгрузить таблицу в CSV/XLSX;
      • "Сгенерировать" — построить типовую кривую для выбранной мощности;
      • "Добавить" — accept() и потом get_diesel_unit() создает DieselUnit.

    Важный момент:
    — Диалог сам НЕ закрывается при get_diesel_unit(); закрытие делает accept() по кнопке OK,
      а объект потом забирает родитель.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавить ДЭУ")
        self.setModal(True)
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        # -------------------------
        # 1) Основные параметры ДЭУ
        # -------------------------
        form_layout = QFormLayout()

        # Название установки (строка). Если оставить пустым — позже подставим default.
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например, ДЭУ-500")

        # Номинальная мощность, кВт. Диапазон широкий, чтобы покрыть учебные и реальные случаи.
        self.nominal_power_spin = QDoubleSpinBox()
        self.nominal_power_spin.setRange(10, 10000)
        self.nominal_power_spin.setSuffix(" кВт")
        self.nominal_power_spin.setValue(500)

        # КПД.
        self.efficiency_spin = QDoubleSpinBox()
        self.efficiency_spin.setRange(0.1, 1.0)
        self.efficiency_spin.setSingleStep(0.01)
        # self.efficiency_spin.setValue(0.85)
        self.efficiency_spin.setSuffix(" %")

        form_layout.addRow("Наименование:", self.name_edit)
        form_layout.addRow("Номинальная мощность:", self.nominal_power_spin)
        form_layout.addRow("КПД:", self.efficiency_spin)

        layout.addLayout(form_layout)

        # -------------------------
        # 2) Блок "Кривая расхода топлива"
        # -------------------------
        curve_group = QGroupBox("Кривая расхода топлива")
        curve_layout = QVBoxLayout(curve_group)

        # Таблица: 2 колонки
        # — Нагрузка, % (0..100)
        # — Расход, л/ч (абсолютный расход при этой нагрузке)
        self.fuel_curve_table = QTableWidget()
        self.fuel_curve_table.setColumnCount(2)
        self.fuel_curve_table.setHorizontalHeaderLabels(["Нагрузка, %", "Расход, л/ч"])
        self.fuel_curve_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Изначально 5 строк — типовой набор точек.
        self.fuel_curve_table.setRowCount(5)

        # Заполняем "типовыми" значениями по умолчанию, чтобы пользователь видел пример формата.
        typical_loads = [0, 25, 50, 75, 100]
        typical_consumption = [0, 70, 120, 180, 250]

        for i, (load, cons) in enumerate(zip(typical_loads, typical_consumption)):
            self.fuel_curve_table.setItem(i, 0, QTableWidgetItem(str(load)))
            self.fuel_curve_table.setItem(i, 1, QTableWidgetItem(str(cons)))

        curve_layout.addWidget(self.fuel_curve_table)

        # -------------------------
        # 3) Кнопки управления кривой
        # -------------------------
        curve_buttons_layout = QHBoxLayout()
        self.load_curve_btn = QPushButton("Загрузить из файла")
        self.save_curve_btn = QPushButton("Сохранить в файл")
        self.generate_curve_btn = QPushButton("Сгенерировать")

        curve_buttons_layout.addWidget(self.load_curve_btn)
        curve_buttons_layout.addWidget(self.save_curve_btn)
        curve_buttons_layout.addWidget(self.generate_curve_btn)
        curve_layout.addLayout(curve_buttons_layout)

        layout.addWidget(curve_group)

        # -------------------------
        # 4) Кнопки диалога
        # -------------------------
        dialog_buttons = QHBoxLayout()
        self.ok_btn = QPushButton("Добавить")
        self.cancel_btn = QPushButton("Отмена")

        # Простая "псевдо-темизация" кнопок через QSS.
        # (если у тебя единый стиль приложения — лучше выкинуть отсюда и задать глобально)
        self.ok_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")

        dialog_buttons.addWidget(self.ok_btn)
        dialog_buttons.addWidget(self.cancel_btn)
        layout.addLayout(dialog_buttons)

        # -------------------------
        # 5) Сигналы/слоты (Qt callbacks)
        # -------------------------
        self.load_curve_btn.clicked.connect(self.load_fuel_curve)
        self.save_curve_btn.clicked.connect(self.save_fuel_curve)
        self.generate_curve_btn.clicked.connect(self.generate_fuel_curve)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def load_fuel_curve(self):
        """
        Загрузка кривой расхода из файла (XLSX/CSV).

        Допущения:
        — файл должен содержать две колонки (любые заголовки),
          и мы берем "первую" как нагрузку (%), "вторую" как расход (л/ч).
        — если у Excel/CSV другой порядок/формат — пользователь получит ошибку/мусор.

        Практическое улучшение:
        — поддержать заголовки 'Load_%' и 'Consumption_l/h' (как в save_fuel_curve),
          и если они есть — читать именно их. Это резко снизит вероятность ошибки.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить кривую расхода", "",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)

                # Перерисовываем таблицу по числу строк в файле
                self.fuel_curve_table.setRowCount(len(df))
                for i, row in df.iterrows():
                    self.fuel_curve_table.setItem(i, 0, QTableWidgetItem(str(row.iloc[0])))
                    self.fuel_curve_table.setItem(i, 1, QTableWidgetItem(str(row.iloc[1])))

                QMessageBox.information(self, "Успех", "Кривая расхода загружена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл: {str(e)}")

    def save_fuel_curve(self):
        """
        Сохранение кривой расхода в файл.

        Формат:
        — DataFrame с колонками ['Load_%', 'Consumption_l/h'].
        Это означает:
        • Load_% — нагрузка в процентах (0..100);
        • Consumption_l/h — расход в л/ч.

        Замечание:
        — Внутри get_diesel_unit() мы преобразуем Load_% в кВт (абсолютную нагрузку),
          потому что core.models.DieselUnit.fuel_curve интерполируется по 'load' (кВт).
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить кривую расхода", "fuel_curve.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if file_path:
            try:
                data = []
                for i in range(self.fuel_curve_table.rowCount()):
                    load_item = self.fuel_curve_table.item(i, 0)
                    cons_item = self.fuel_curve_table.item(i, 1)
                    if load_item and cons_item:
                        data.append([float(load_item.text()), float(cons_item.text())])

                df = pd.DataFrame(data, columns=['Load_%', 'Consumption_l/h'])
                if file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Успех", "Кривая расхода сохранена")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл: {str(e)}")

    def generate_fuel_curve(self):
        """
        Генерация "типовой" кривой расхода топлива.

        Вход:
        — nominal_power (кВт).

        Эвристика:
        — typical_consumption задан как л/кВт*ч для разных уровней нагрузки.
          Затем переводим в л/ч: (л/кВт*ч) * (кВт) = л/ч
        — При нагрузке L% мощность ~ nominal_power * L/100.

        Важно:
        — Это не "реальная" модель двигателя, а шаблон для заполнения.
        — Пользователь должен иметь возможность поправить точки вручную.
        """
        nominal_power = self.nominal_power_spin.value()

        loads = [0, 25, 50, 75, 100]
        # Значения — условные. Можно сделать эти коэффициенты параметрическими.
        typical_consumption = [0, 0.4, 0.7, 1.0, 1.2]  # л/кВт*ч

        consumptions = [cons * nominal_power * load / 100 for load, cons in zip(loads, typical_consumption)]

        self.fuel_curve_table.setRowCount(len(loads))
        for i, (load, cons) in enumerate(zip(loads, consumptions)):
            self.fuel_curve_table.setItem(i, 0, QTableWidgetItem(str(load)))
            self.fuel_curve_table.setItem(i, 1, QTableWidgetItem(f"{cons:.1f}"))

    def get_diesel_unit(self):
        """
        Сформировать DieselUnit из значений UI.

        Логика конвертации:
        — В таблице пользователь вводит нагрузку в процентах.
        — В модели curve.x ожидается нагрузка в кВт (абсолютная мощность).
          Поэтому:
            load_kw = load_pct/100 * nominal_power
        — consumption остается в л/ч (абсолютный расход).

        Валидация "минимально достаточная":
        — если в ячейке мусор (не число) — строку пропускаем.
        — если таблица пустая — fuel_curve будет пустым DataFrame, а fuel_consumption вернет 0.
          (но по-хорошему надо предупреждать пользователя)
        """
        name = self.name_edit.text() or f"ДЭУ-{self.nominal_power_spin.value()}"
        nominal_power = self.nominal_power_spin.value()
        efficiency = self.efficiency_spin.value()

        fuel_curve_data = []
        for i in range(self.fuel_curve_table.rowCount()):
            load_item = self.fuel_curve_table.item(i, 0)
            cons_item = self.fuel_curve_table.item(i, 1)
            if load_item and cons_item:
                try:
                    load_pct = float(load_item.text())
                    consumption = float(cons_item.text())
                    fuel_curve_data.append({
                        'load': load_pct / 100 * nominal_power,  # кВт (абсолютная нагрузка)
                        'consumption': consumption               # л/ч
                    })
                except ValueError:
                    continue

        fuel_curve = pd.DataFrame(fuel_curve_data)

        return DieselUnit(
            name=name,
            nominal_power=nominal_power,
            efficiency=efficiency,
            fuel_curve=fuel_curve
        )


class DieselPlantDialog(QDialog):
    """
    Диалог создания ДЭС как набора ДЭУ (DieselPlant).

    Вход:
    — diesel_units: список доступных DieselUnit (обычно из каталога/ранее созданных).

    UI:
    — поле имени станции;
    — список DЭУ с чекбоксами;
    — OK/Cancel.

    Выход:
    — get_diesel_plant() собирает выбранные DieselUnit и возвращает DieselPlant.
    """

    def __init__(self, diesel_units, parent=None):
        super().__init__(parent)
        self.diesel_units = diesel_units
        self.setWindowTitle("Создать ДЭС")
        self.setModal(True)
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # Название ДЭС
        form_layout = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Например, ДЭС-Основная")
        form_layout.addRow("Название ДЭС:", self.name_edit)
        layout.addLayout(form_layout)

        layout.addWidget(QLabel("Выберите ДЭУ для включения в ДЭС:"))

        # Список с checkable item'ами. В item.data(Qt.UserRole) прячем ссылку на объект DieselUnit.
        self.units_list = QListWidget()
        for unit in diesel_units:
            item = QListWidgetItem(f"{unit.name} ({unit.nominal_power} кВт)")
            item.setData(Qt.UserRole, unit)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.units_list.addItem(item)

        layout.addWidget(self.units_list)

        # Кнопки диалога
        buttons_layout = QHBoxLayout()
        self.ok_btn = QPushButton("Создать ДЭС")
        self.cancel_btn = QPushButton("Отмена")

        self.ok_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        self.cancel_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")

        buttons_layout.addWidget(self.ok_btn)
        buttons_layout.addWidget(self.cancel_btn)
        layout.addLayout(buttons_layout)

        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_diesel_plant(self):
        """
        Собрать DieselPlant из выбранных чекбоксов.

        Важно:
        — если ничего не выбрано, вернется DieselPlant с пустым списком агрегатов.
          Это, скорее всего, ошибка ввода. В реальном UI лучше:
          • блокировать OK, пока не выбран хотя бы один агрегат;
          • или показать предупреждение, если selected_units пуст.
        """
        name = self.name_edit.text() or "ДЭС"

        selected_units = []
        for i in range(self.units_list.count()):
            item = self.units_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_units.append(item.data(Qt.UserRole))

        return DieselPlant(name=name, diesel_units=selected_units)


class HydroPlantDialog(QDialog):
    """
    Диалог добавления МГЭС (HydroPlant) с обязательным гидрографом.

    Что вводим:
    — name: строка;
    — nominal_power: кВт;
    — efficiency: КПД (0..1);
    — head: напор (м);
    — hydro_graph: словарь расходов по месяцам (м^3/с).

    Почему гидрограф "обязателен":
    — без него HydroPlant.available_power() либо будет работать по "50% от номинала" (fallback),
      либо даст несопоставимые результаты (зависит от твоей логики).
    — UI сейчас жестко требует загрузку: OK заблокирована, пока hydro_graph не загружен.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавление МГЭС")
        self.setGeometry(300, 300, 500, 400)

        self.layout = QVBoxLayout(self)

        # -------------------------
        # 1) Основные поля ввода
        # -------------------------
        form_layout = QFormLayout()

        self.name_edit = QLineEdit("МГЭС")

        self.power_edit = QDoubleSpinBox()
        self.power_edit.setRange(10, 10000)
        self.power_edit.setValue(1000)
        self.power_edit.setSuffix(" кВт")
        self.power_edit.setSingleStep(100)

        self.efficiency_edit = QDoubleSpinBox()
        self.efficiency_edit.setRange(0.1, 1.0)
        self.efficiency_edit.setValue(0.85)
        self.efficiency_edit.setSingleStep(0.05)
        self.efficiency_edit.setDecimals(2)

        self.head_edit = QDoubleSpinBox()
        self.head_edit.setRange(1, 500)
        self.head_edit.setValue(50)
        self.head_edit.setSuffix(" м")
        self.head_edit.setSingleStep(5)

        form_layout.addRow("Название:", self.name_edit)
        form_layout.addRow("Номинальная мощность:", self.power_edit)
        form_layout.addRow("КПД:", self.efficiency_edit)
        form_layout.addRow("Напор:", self.head_edit)

        # -------------------------
        # 2) Блок загрузки гидрографа
        # -------------------------
        self.hydro_graph_group = QGroupBox("Гидрограф (обязательно)")
        hydro_graph_layout = QVBoxLayout(self.hydro_graph_group)

        self.load_hydro_graph_btn = QPushButton("Загрузить гидрограф из Excel")
        self.load_hydro_graph_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Статус и краткая статистика по гидрографу
        self.hydro_graph_status = QLabel("❌ Гидрограф не загружен")
        self.hydro_graph_status.setStyleSheet("font-weight: bold; color: #d32f2f;")

        self.hydro_graph_info = QLabel("")
        self.hydro_graph_info.setStyleSheet("color: #666; font-size: 10px;")

        hydro_graph_layout.addWidget(self.load_hydro_graph_btn)
        hydro_graph_layout.addWidget(self.hydro_graph_status)
        hydro_graph_layout.addWidget(self.hydro_graph_info)

        # hydro_graph хранится как dict[int, float] после загрузки:
        # {1: Q1, 2: Q2, ..., 12: Q12}
        self.hydro_graph = None

        # -------------------------
        # 3) Стандартные кнопки OK/Cancel
        # -------------------------
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

        # OK выключена, пока гидрограф не загружен (жесткая валидация).
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        # Компоновка
        self.layout.addLayout(form_layout)
        self.layout.addWidget(self.hydro_graph_group)
        self.layout.addWidget(self.button_box)

        # Сигналы/слоты
        self.load_hydro_graph_btn.clicked.connect(self.load_hydro_graph)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def load_hydro_graph(self):
        """
        Загрузка гидрографа из Excel.

        Поддерживаемые форматы (по твоей логике):
        1) "Ваш формат" (2 строки, 13 столбцов):
           — строка 0: [текст, 1,2,3,...,12]
           — строка 1: [текст, Q1,Q2,...,Q12]
        2) Два столбца:
           — столбец 0: месяц (1..12)
           — столбец 1: расход (м^3/с)
        3) Фоллбек-парсинг:
           — собираем все числовые значения из файла;
           — ищем подряд 12 чисел (0..10000) и считаем, что это расходы по месяцам.

        После успеха:
        — self.hydro_graph = dict;
        — обновляются статусы/статистика;
        — OK становится доступной.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Загрузить гидрограф МГЭС", "",
                "Excel Files (*.xlsx *.xls);;All Files (*)"
            )
            if not file_path:
                return

            df = pd.read_excel(file_path, header=None)
            hydro_graph = {}

            # Формат 1: 2 строки, 13 столбцов (первый столбец — подпись)
            if df.shape[0] >= 2 and df.shape[1] >= 13:
                for col in range(1, 13):
                    if col < df.shape[1]:
                        try:
                            month_cell = df.iloc[0, col]
                            month = int(month_cell) if not pd.isna(month_cell) else col

                            flow_cell = df.iloc[1, col]
                            flow = float(flow_cell) if not pd.isna(flow_cell) else 0.0

                            hydro_graph[month] = flow
                        except Exception:
                            hydro_graph[col] = 0.0

            # Формат 2: два столбца
            if not hydro_graph and df.shape[1] >= 2:
                for i in range(min(12, df.shape[0])):
                    try:
                        month = int(df.iloc[i, 0])
                        flow = float(df.iloc[i, 1])
                        if 1 <= month <= 12:
                            hydro_graph[month] = flow
                    except Exception:
                        continue

            # Формат 3: "поиск 12 чисел"
            if not hydro_graph:
                all_numbers = []
                for i in range(df.shape[0]):
                    for j in range(df.shape[1]):
                        try:
                            val = df.iloc[i, j]
                            if not pd.isna(val):
                                all_numbers.append(float(val))
                        except Exception:
                            continue

                for i in range(len(all_numbers) - 11):
                    potential_flows = all_numbers[i:i + 12]
                    if all(0 <= f <= 10000 for f in potential_flows):
                        for month, flow in enumerate(potential_flows, 1):
                            hydro_graph[month] = flow
                        break

            # Валидация результата
            if not hydro_graph:
                raise ValueError("Не удалось найти данные гидрографа в файле")

            # Гарантируем 1..12
            for month in range(1, 13):
                if month not in hydro_graph:
                    hydro_graph[month] = 0.0

            total_flow = sum(hydro_graph.values())
            if total_flow < 0.1:
                raise ValueError(
                    f"Загруженные данные близки к нулю (сумма: {total_flow}). Проверьте формат файла."
                )

            self.hydro_graph = hydro_graph

            # Статистика для UI
            min_flow = min(hydro_graph.values())
            max_flow = max(hydro_graph.values())
            avg_flow = total_flow / len(hydro_graph)

            self.hydro_graph_status.setText("✅ Гидрограф загружен успешно")
            self.hydro_graph_status.setStyleSheet("font-weight: bold; color: #388e3c;")

            self.hydro_graph_info.setText(
                f"Месяцы: {len(hydro_graph)}/12 | "
                f"Мин: {min_flow:.0f} м³/с | "
                f"Макс: {max_flow:.0f} м³/с | "
                f"Средн: {avg_flow:.0f} м³/с"
            )

            # Детализация в сообщении (удобно для контроля от студента)
            details = "Загруженные данные:\n"
            for month in sorted(hydro_graph.keys()):
                details += f"Месяц {month}: {hydro_graph[month]:.0f} м³/с\n"

            # Разрешаем OK
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)

            QMessageBox.information(
                self, "Успех",
                f"Гидрограф успешно загружен!\n"
                f"Загружено {len(hydro_graph)} месяцев\n"
                f"Средний расход: {avg_flow:.0f} м³/с\n\n"
                f"{details}"
            )

        except Exception as e:
            # При ошибке показываем "человеческий" формат и подсказку про допустимые форматы.
            error_msg = f"Ошибка загрузки гидрографа: {str(e)}\n\n"
            error_msg += "Формат файла должен быть одним из:\n"
            error_msg += "1. Две строки:\n"
            error_msg += "   - Первая: 'Месяц' затем числа 1-12\n"
            error_msg += "   - Вторая: 'Среднемесячный расход' затем расходы для каждого месяца\n"
            error_msg += "2. Два столбца:\n"
            error_msg += "   - Первый: месяцы 1-12\n"
            error_msg += "   - Второй: расходы для каждого месяца"

            QMessageBox.critical(self, "Ошибка", error_msg)

    def get_hydro_plant(self):
        """
        Создать объект HydroPlant из значений UI.

        Жесткая валидация:
        — если гидрограф не загружен, бросаем исключение.
          (в нормальном UI до этого не дойдет, потому что OK заблокирована)
        """
        if not self.hydro_graph:
            raise ValueError("Гидрограф не загружен")

        return HydroPlant(
            name=self.name_edit.text(),
            nominal_power=self.power_edit.value(),
            efficiency=self.efficiency_edit.value(),
            head=self.head_edit.value(),
            hydro_graph=self.hydro_graph
        )
