# File: main.py
"""
Точка входа приложения + "контроллер" уровня приложения (El Despacho Maestro) + поток оптимизации (OptimizationThread).

Этот файл отвечает за:
1) запуск Qt-приложения и создание GUI (MainWindow);
2) подключение сигналов кнопок UI к методам (setup_connections);
3) хранение "живых" объектов данных (каталоги оборудования, профили нагрузки/ветра, результаты оптимизации);
4) запуск оптимизации в отдельном QThread, чтобы UI не зависал;
5) обновление графиков/таблиц по завершении оптимизации;
6) синхронизацию выбора даты (QDateEdit) с данными schedule/load_profile.

⚠️ Важно про архитектуру:
• ui/main_window.py — чистый View (создает виджеты, умеет показывать данные).
• main.py (EnergyOptimizerApp) — Controller/Coordinator:
  — знает, откуда брать данные и куда их класть;
  — знает логику "что делать при нажатии кнопки";
  — создает поток оптимизации, получает результат и обновляет UI.
• core/* — бизнес-логика и модели (OptimizationEngine, DataLoader, модели оборудования, графики).

⚠️ Про потоки:
• OptimizationThread выполняет тяжелый расчет в фоне.
• Любые обращения к реальным виджетам Qt должны происходить в главном потоке.
  Здесь это соблюдается: поток только emit() сигнал finished/error.
  Прогресс идет через progress_cb = self.main_window._on_progress:
  — это потенциально тонкий момент, потому что progress_cb вызывается из потока.
  В текущей реализации _on_progress только пишет в поля (self._progress_state),
  и UI обновляется таймером в главном потоке -> обычно это ок.
  Но строгий Qt-way: прогресс лучше эмитить signal и ловить его в main thread.

"""

import sys


def _velopack_bootstrap() -> None:
    """
    Должно быть самым первым.
    На install/update/uninstall Velopack сам обработает событие и выйдет,
    а GUI не будет подниматься.
    """
    try:
        from velopack import App
        App().run()
    except Exception:
        pass

_velopack_bootstrap()


import os

import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMessageBox, QFileDialog, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtGui import QPalette, QColor

# ---- Импорт доменной логики и UI ----
from core.models import (
    DieselUnit, DieselPlant, HydroPlant, WindTurbine, Battery,
    LoadProfile, OptimizationResult
)
from core.optimization import OptimizationEngine
from core.data_loader import DataLoader
from ui.main_window import MainWindow
from ui.equipment_dialogs import DieselUnitDialog, DieselPlantDialog, HydroPlantDialog


class OptimizationThread(QThread):
    """
    Поток оптимизации.

    Зачем нужен:
    • OptimizationEngine.optimize(...) потенциально выполняется долго (8760 часов, варианты состава, GWO/DE),
      и если вызывать его прямо из обработчика кнопки — UI "повиснет".
    • QThread позволяет выполнить расчет в фоне и уведомить UI по завершении.

    Сигналы:
    • finished(object) — возвращаем объект OptimizationResult (или совместимый тип).
    • error(str) — текст ошибки, если в потоке вылетело исключение.
    • progress(int) — объявлен, но В ДАННОМ КОДЕ НЕ ИСПОЛЬЗУЕТСЯ
      (фактический прогресс идет через progress_cb, который уходит внутрь optimize()).

    ⚠️ Важно:
    Поток НЕ ДОЛЖЕН напрямую трогать GUI (виджеты Qt), иначе будут рандомные краши.
    """
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int)  # сейчас не используется

    def __init__(self, optimization_params: dict):
        super().__init__()
        # params — единый словарь, чтобы не плодить много аргументов
        self.params = optimization_params

    def run(self):
        """
        Основная функция потока.
        Здесь выполняется оптимизация и отправляется результат сигналом finished.
        Любая ошибка заворачивается в error.emit.
        """
        try:
            # -------------------------
            # 1) Извлекаем параметры
            # -------------------------
            load_profile = self.params.get('load_profile')
            diesel_plants = self.params.get('diesel_plants', [])
            hydro_plants = self.params.get('hydro_plants', [])
            wind_turbines = self.params.get('wind_turbines', [])
            batteries = self.params.get('batteries', [])
            algorithm = self.params.get('algorithm', 'greedy')
            wind_data = self.params.get('wind_data')
            progress_cb = self.params.get('progress_cb')

            # -------------------------
            # 2) Примитивная валидация входов
            # -------------------------
            # Эти проверки дополнительно дублируют validate_inputs(), но это нормально:
            # поток должен быть "самозащищенным", потому что его могут вызвать некорректно.
            if load_profile is None:
                raise ValueError("Не загружен график нагрузки")
            if not diesel_plants:
                raise ValueError("Не добавлены ДЭС")

            # -------------------------
            # 3) Обработка wind_data в удобный для оптимизатора формат
            # -------------------------
            # Твоя оптимизация ожидает wind_speeds: List[float] или None.
            wind_speeds = None
            if wind_data is not None:
                # DataFrame-подобный объект (pandas)
                if hasattr(wind_data, 'empty'):
                    if not wind_data.empty:
                        # ожидаем колонку 'wind_speed'
                        wind_speeds = wind_data['wind_speed'].tolist()
                # Если кто-то уже передал список
                elif isinstance(wind_data, list):
                    wind_speeds = wind_data
                # Иначе — неизвестный формат, считаем что данных нет.

            print(f"Запуск оптимизации: алгоритм={algorithm}, часов={len(load_profile.data)}")

            # -------------------------
            # 4) Запуск ядра оптимизации
            # -------------------------
            # Важно: optimize(...) должен быть потокобезопасным (не дергать UI).
            result = OptimizationEngine.optimize(
                load_profile=load_profile,
                diesel_plants=diesel_plants,
                hydro_plants=hydro_plants,
                wind_turbines=wind_turbines,
                batteries=batteries,
                algorithm=algorithm,
                wind_speeds=wind_speeds,
                progress_cb=progress_cb,
            )

            # -------------------------
            # 5) Возвращаем результат
            # -------------------------
            self.finished.emit(result)

        except Exception as e:
            # Печать полного стека помогает дебагу при падениях в потоке
            import traceback
            error_details = traceback.format_exc()
            print(f"Полная ошибка оптимизации:\n{error_details}")

            # В UI отправляем короткое сообщение без стека
            self.error.emit(f"Ошибка оптимизации: {str(e)}")


class EnergyOptimizerApp:
    """
    "Контроллер" приложения.

    Жизненный цикл:
    1) создать QApplication;
    2) применить тему;
    3) создать MainWindow (View);
    4) инициализировать состояния (каталоги, профили, результата);
    5) подключить сигналы UI к методам;
    6) показать окно и запустить Qt-цикл.

    Здесь хранятся:
    • каталоги оборудования (diesel_units, diesel_plants, hydro_plants, wind_turbines, batteries);
    • входные данные (load_profile, wind_data);
    • результат (optimization_result);
    • поток оптимизации (optimization_thread).
    """

    def __init__(self):
        # -------------------------
        # 1) Создание Qt-приложения и темы
        # -------------------------
        self.app = QApplication(sys.argv)
        self.set_dark_theme()

        # -------------------------
        # 2) Создаем главное окно (UI View)
        # -------------------------
        self.main_window = MainWindow()

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
        from core.updater import check_for_updates, download_and_apply_update

        self.update_feed_url = "https://github.com/H2K2U/El-Despacho-Maestro"  # потом заменишь на реальный

        def _maybe_update():
            try:
                res = check_for_updates(self.update_feed_url)
                if not res.has_update:
                    return
                reply = QMessageBox.question(
                    self.main_window,
                    "Доступно обновление",
                    "Найдена новая версия. Скачать и перезапустить программу?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes and res.update_info:
                    download_and_apply_update(self.update_feed_url, res.update_info)
            except Exception as e:
                # Обновления не должны ломать запуск приложения
                print(f"Update check failed: {e}")

        QTimer.singleShot(1500, _maybe_update)

        # -------------------------
        # 3) Центрируем окно
        # -------------------------
        # Это purely UX: появляется по центру экрана.
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = self.main_window.frameGeometry()
        window_geometry.moveCenter(screen_geometry.center())
        self.main_window.move(window_geometry.topLeft())

        # -------------------------
        # 4) Сигналы/слоты
        # -------------------------
        self.setup_connections()

        # -------------------------
        # 5) Инициализация данных состояния
        # -------------------------
        self.diesel_units = []
        self.diesel_plants = []
        self.hydro_plants = []
        self.wind_turbines = []
        self.batteries = []

        self.load_profile = None      # типично LoadProfile (обертка над df)
        self.wind_data = None         # типично DataFrame с 'wind_speed'
        self.optimization_result = None  # OptimizationResult

        self.optimization_thread = None

        # Устанавливаем диапазон дат сразу (пока fallback 2021),
        # затем он будет обновляться после загрузки данных.
        self.update_date_selector_range()

    def set_dark_theme(self):
        """
        Устанавливает темную тему на уровне QApplication через QPalette.

        Почему именно QPalette:
        • в Fusion-стиле это самый прямой способ задать "темный режим" на все виджеты.
        • затем локально некоторые элементы (кнопки) дополнительно стилятся через setStyleSheet.

        Важно:
        • если потом добавятся кастомные стили (QSS), они могут конфликтовать с палитрой.
        """
        self.app.setStyle('Fusion')
        palette = QPalette()

        # Фон окна и текст
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)

        # Тело виджетов (фон полей ввода/таблиц)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))

        # Альтернативный фон (например, в таблицах для чередования строк)
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))

        # Tooltips
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)

        # Текст/кнопки
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)

        # Акцентные цвета
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
        palette.setColor(QPalette.HighlightedText, Qt.black)

        self.app.setPalette(palette)

    def setup_connections(self):
        """
        Подключение сигналов UI к методам контроллера.

        Это "проводка" между View и логикой.

        Список:
        • кнопки загрузки -> load_* методы;
        • кнопки добавления оборудования -> диалоги -> обновление списков;
        • optimize_btn -> run_optimization;
        • dateChanged -> update_visualization (перерисовка);
        """
        self.main_window.load_load_profile_btn.clicked.connect(self.load_load_profile)
        self.main_window.add_diesel_unit_btn.clicked.connect(self.add_diesel_unit)
        self.main_window.create_diesel_plant_btn.clicked.connect(self.create_diesel_plant)
        self.main_window.add_hydro_plant_btn.clicked.connect(self.add_hydro_plant)
        self.main_window.load_wind_turbines_btn.clicked.connect(self.load_wind_turbines)
        self.main_window.load_batteries_btn.clicked.connect(self.load_batteries)
        self.main_window.optimize_btn.clicked.connect(self.run_optimization)
        self.main_window.date_selector.dateChanged.connect(self.update_visualization)
        self.main_window.load_wind_data_btn.clicked.connect(self.load_wind_data)

    # -------------------------------------------------------------------------
    # ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------------------------------
    def load_hydro_graph(self):
        """
        Загрузка гидрографа для МГЭС.

        ⚠️ В текущей версии это "старый" метод: теперь гидрограф грузится внутри HydroPlantDialog.
        Поэтому здесь несколько проблем:
        • self.main_window.hydro_graph_status — такого поля нет в MainWindow (оно в диалоге HydroPlantDialog).
        • гидрограф (self.hydro_graph) не используется системно.
        • метод не подключен ни к одной кнопке.

        Вывод:
        — метод можно удалить или переделать под новую архитектуру.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Загрузить гидрограф МГЭС", "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            if file_path:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path, encoding='utf-8')

                hydro_graph = {}

                if 'Месяц' in df.columns and 'Среднемесячный расход' in df.columns:
                    for _, row in df.iterrows():
                        month = int(row['Месяц'])
                        flow = float(row['Среднемесячный расход'])
                        hydro_graph[month] = flow
                elif df.shape[1] >= 13:
                    for i in range(1, 13):
                        if i < len(df.columns):
                            try:
                                hydro_graph[i] = float(df.iloc[0, i])
                            except Exception:
                                hydro_graph[i] = 0.0

                self.hydro_graph = hydro_graph

                # Прокидываем загруженный гидрограф во все уже созданные МГЭС
                for hydro_plant in self.hydro_plants:
                    if hasattr(hydro_plant, 'hydro_graph'):
                        hydro_plant.hydro_graph = hydro_graph

                # ⚠️ потенциальный AttributeError: такого label нет в MainWindow
                self.main_window.hydro_graph_status.setText(f"Загружен: {len(hydro_graph)} месяцев")

                QMessageBox.information(
                    self.main_window, "Успех",
                    f"Гидрограф загружен: {len(hydro_graph)} месяцев\n"
                    f"Пример: январь={hydro_graph.get(1, 0):.0f} м³/с"
                )

        except Exception as e:
            QMessageBox.critical(
                self.main_window, "Ошибка",
                f"Ошибка загрузки гидрографа: {str(e)}\n\n"
                f"Ожидаемый формат:\n"
                f"Столбцы: 'Месяц', 'Среднемесячный расход'\n"
                f"Или 12 столбцов с расходами по месяцам"
            )

    def load_wind_data(self):
        """
        Загрузка временного ряда скорости ветра.

        Ожидается:
        • Excel-файл с колонками, которые DataLoader преобразует в DataFrame
          с обязательной колонкой 'wind_speed' (см. OptimizationThread.run).

        Алгоритм:
        1) QFileDialog -> выбрать файл;
        2) DataLoader.load_wind_data(file_path) -> DataFrame;
        3) обновить статус и показать QMessageBox.

        Ошибки оборачиваются в удобное сообщение для пользователя
        с объяснением требований к формату.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Загрузить данные о ветре", "",
                "Excel Files (*.xlsx);;All Files (*)"
            )
            if file_path:
                self.wind_data = DataLoader.load_wind_data(file_path)

                if self.wind_data is not None and not self.wind_data.empty:
                    self.main_window.wind_data_status.setText(f"Загружены: {len(self.wind_data)} записей")
                    QMessageBox.information(self.main_window, "Успех", "Данные о ветре успешно загружены")
                else:
                    self.wind_data = None
                    self.main_window.wind_data_status.setText("Ошибка загрузки")
                    QMessageBox.warning(
                        self.main_window, "Предупреждение",
                        "Файл ветровых данных пуст или содержит ошибки"
                    )
        except Exception as e:
            self.wind_data = None
            error_msg = f"Ошибка загрузки: {str(e)}\n\n"
            error_msg += "Проверьте формат файла:\n"
            error_msg += "- Должны быть колонки 'Время' и 'Скорость'\n"
            error_msg += "- Время в формате 'dd.mm.yyyy hh:mm'\n"
            error_msg += "- Числа с запятой как разделителем дробной части"
            QMessageBox.critical(self.main_window, "Ошибка", error_msg)

    def load_load_profile(self):
        """
        Загрузка профиля нагрузки (график нагрузки).

        Ожидается:
        • Excel/CSV, который DataLoader приводит к LoadProfile,
          где внутри лежит DataFrame (load_profile.data)
          и индекс часто DatetimeIndex (если DataLoader так сделал).

        После успешной загрузки:
        • обновляется статус слева;
        • пересчитывается диапазон дат для date_selector (set_date_range);
        • показывается информационное окно.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Загрузить график нагрузки", "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            if file_path:
                # Сейчас оба случая ведут к одному методу; условие оставлено как "наглядность".
                self.load_profile = DataLoader.load_load_profile(file_path)

                if self.load_profile is not None and not self.load_profile.data.empty:
                    self.main_window.load_profile_status.setText(f"Загружен: {len(self.load_profile.data)} записей")
                    self.update_date_selector_range()
                    QMessageBox.information(self.main_window, "Успех", "График нагрузки успешно загружен")
                else:
                    self.load_profile = None
                    self.main_window.load_profile_status.setText("Ошибка загрузки")
                    QMessageBox.warning(
                        self.main_window, "Предупреждение",
                        "Файл графика нагрузки пуст или содержит ошибки"
                    )
        except Exception as e:
            self.load_profile = None
            error_msg = f"Ошибка загрузки: {str(e)}\n\n"
            error_msg += "Проверьте формат файла:\n"
            error_msg += "- Должны быть колонки 'Время' и 'Потребление'\n"
            error_msg += "- Время в формате 'dd.mm.yyyy hh:mm'\n"
            error_msg += "- Числа с запятой как разделителем дробной части"
            QMessageBox.critical(self.main_window, "Ошибка", error_msg)

    # -------------------------------------------------------------------------
    # РАБОТА С ОБОРУДОВАНИЕМ (через диалоги)
    # -------------------------------------------------------------------------
    def add_diesel_unit(self):
        """
        Добавить одну ДЭУ.

        Как это работает:
        1) открываем DieselUnitDialog;
        2) если пользователь нажал "Добавить" (Accepted):
           — читаем данные dialog.get_diesel_unit();
           — добавляем объект DieselUnit в self.diesel_units;
           — обновляем списки на левой панели.
        """
        dialog = DieselUnitDialog(self.main_window)
        if dialog.exec() == QDialog.Accepted:
            diesel_unit = dialog.get_diesel_unit()
            self.diesel_units.append(diesel_unit)
            self.update_equipment_lists()
            QMessageBox.information(self.main_window, "Успех", f"ДЭУ '{diesel_unit.name}' добавлена")

    def create_diesel_plant(self):
        """
        Создать ДЭС из выбранных ДЭУ.

        Логика:
        • если diesel_units пуст — нельзя создать станцию;
        • DieselPlantDialog дает список с чекбоксами;
        • результат dialog.get_diesel_plant() содержит список diesel_units внутри;
        • если пользователь ничего не выбрал — предупреждаем.
        """
        if not self.diesel_units:
            QMessageBox.warning(self.main_window, "Предупреждение", "Сначала добавьте хотя бы одну ДЭУ")
            return

        dialog = DieselPlantDialog(self.diesel_units, self.main_window)
        if dialog.exec() == QDialog.Accepted:
            diesel_plant = dialog.get_diesel_plant()

            if not diesel_plant.diesel_units:
                QMessageBox.warning(self.main_window, "Предупреждение", "Выберите хотя бы одну ДЭУ для ДЭС")
                return

            self.diesel_plants.append(diesel_plant)
            self.update_equipment_lists()
            QMessageBox.information(self.main_window, "Успех", f"ДЭС '{diesel_plant.name}' создана")

    def add_hydro_plant(self):
        """
        Добавить МГЭС через HydroPlantDialog.

        Особенность:
        • гидрограф обязателен (в диалоге кнопка OK заблокирована, пока гидрограф не загружен).

        После добавления:
        • обновляются списки;
        • показывается окно с резюме параметров и статистикой гидрографа.
        """
        dialog = HydroPlantDialog(self.main_window)
        if dialog.exec() == QDialog.Accepted:
            try:
                hydro_plant = dialog.get_hydro_plant()
                self.hydro_plants.append(hydro_plant)
                self.update_equipment_lists()

                if hydro_plant.hydro_graph:
                    months = len(hydro_plant.hydro_graph)
                    avg_flow = sum(hydro_plant.hydro_graph.values()) / months
                    QMessageBox.information(
                        self.main_window, "Успех",
                        f"МГЭС '{hydro_plant.name}' добавлена\n"
                        f"Номинальная мощность: {hydro_plant.nominal_power} кВт\n"
                        f"Загружен гидрограф: {months} месяцев\n"
                        f"Средний расход: {avg_flow:.0f} м³/с"
                    )
                else:
                    # по идее не должно случаться, поскольку OK включается только после загрузки
                    QMessageBox.warning(
                        self.main_window, "Предупреждение",
                        f"МГЭС '{hydro_plant.name}' добавлена без гидрографа!\n"
                        f"Для корректной работы необходимо загрузить гидрограф."
                    )

            except ValueError as e:
                QMessageBox.critical(
                    self.main_window, "Ошибка",
                    f"Ошибка создания МГЭС: {str(e)}\nПожалуйста, загрузите гидрограф."
                )
            except Exception as e:
                QMessageBox.critical(self.main_window, "Ошибка", f"Непредвиденная ошибка: {str(e)}")

    def load_wind_turbines(self):
        """
        Загрузка каталога ВЭУ.

        Важно:
        • DataLoader имеет два варианта парсинга:
          — основной (load_wind_turbines);
          — альтернативный (load_wind_turbines_alternative),
            который используется как fallback, если основной формат не распознался.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Загрузить каталог ВЭУ", "",
                "Excel Files (*.xlsx);;All Files (*)"
            )
            if file_path:
                try:
                    self.wind_turbines = DataLoader.load_wind_turbines(file_path)
                except Exception as e1:
                    print(f"Первый формат не сработал: {e1}")
                    try:
                        self.wind_turbines = DataLoader.load_wind_turbines_alternative(file_path)
                    except Exception as e2:
                        raise ValueError(
                            "Не удалось загрузить файл ВЭУ. Формат не распознан.\n"
                            f"Ошибка 1: {e1}\nОшибка 2: {e2}"
                        )

                self.update_equipment_lists()
                QMessageBox.information(self.main_window, "Успех", f"Загружено {len(self.wind_turbines)} ВЭУ")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Ошибка", f"Ошибка загрузки: {str(e)}")

    def load_batteries(self):
        """
        Загрузка каталога АКБ.

        Вход:
        • Excel/CSV.

        Выход:
        • self.batteries = List[Battery] (объекты моделей).
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window, "Загрузить каталог АКБ", "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
            )
            if file_path:
                self.batteries = DataLoader.load_batteries(file_path)
                self.update_equipment_lists()
                QMessageBox.information(self.main_window, "Успех", f"Загружено {len(self.batteries)} АКБ")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Ошибка", f"Ошибка загрузки: {str(e)}")

    def update_equipment_lists(self):
        """
        Прокси-метод: дергает MainWindow.update_equipment_lists(...)
        чтобы обновить видимые списки оборудования на левой панели.
        """
        self.main_window.update_equipment_lists(
            self.diesel_units, self.diesel_plants, self.hydro_plants,
            self.wind_turbines, self.batteries
        )

    # -------------------------------------------------------------------------
    # ЗАПУСК ОПТИМИЗАЦИИ
    # -------------------------------------------------------------------------
    def run_optimization(self):
        """
        Запустить оптимизацию по текущим входным данным и выбранному алгоритму.

        Шаги:
        1) validate_inputs() — проверяем, что минимум данных есть;
        2) блокируем кнопку, показываем прогресс;
        3) собираем параметры в dict (optimization_params);
        4) создаем OptimizationThread(params) и подключаем finished/error;
        5) стартуем поток.

        ⚠️ Важно:
        Здесь нет подключения самодельного progress(int) сигнала потока.
        Прогресс идет через progress_cb=self.main_window._on_progress.
        """
        if not self.validate_inputs():
            return

        # Блокируем кнопку запуска, чтобы пользователь не запустил второй поток.
        self.main_window.optimize_btn.setEnabled(False)
        self.main_window.progress_bar.setVisible(True)

        # Код алгоритма выбираем из userData combobox.
        algorithm = self.main_window.algorithm_selector.currentData() or "greedy"

        # Стартовые параметры прогресс-бара.
        self.main_window.progress_bar.setRange(0, 100)
        self.main_window.progress_bar.setValue(0)
        self.main_window.progress_bar.setFormat("0% — старт")

        optimization_params = {
            'load_profile': self.load_profile,
            'diesel_plants': self.diesel_plants,
            'hydro_plants': self.hydro_plants,
            'wind_turbines': self.wind_turbines,
            'batteries': self.batteries,
            'wind_data': self.wind_data,
            'algorithm': algorithm,

            # progress_cb будет вызываться из фонового потока.
            # _on_progress только пишет state, UI обновит таймер (в main thread).
            'progress_cb': self.main_window._on_progress,
        }

        self.optimization_thread = OptimizationThread(optimization_params)
        self.optimization_thread.finished.connect(self.on_optimization_finished)
        self.optimization_thread.error.connect(self.on_optimization_error)
        self.optimization_thread.start()

    def validate_inputs(self) -> bool:
        """
        Проверка минимально необходимых входов для запуска оптимизации.

        Требования:
        • load_profile должен быть загружен;
        • хотя бы одна ДЭС должна быть создана (diesel_plants);
        • если загружены ВЭУ, но нет wind_data — спрашиваем подтверждение.

        Возвращает:
        • True — можно стартовать;
        • False — нельзя.
        """
        if not self.load_profile:
            QMessageBox.warning(self.main_window, "Предупреждение", "Сначала загрузите график нагрузки")
            return False

        if not self.diesel_plants:
            QMessageBox.warning(self.main_window, "Предупреждение", "Добавьте хотя бы одну ДЭС")
            return False

        has_wind_turbines = bool(self.wind_turbines) and len(self.wind_turbines) > 0
        has_wind_data = self.wind_data is not None and not self.wind_data.empty

        if has_wind_turbines and not has_wind_data:
            reply = QMessageBox.question(
                self.main_window,
                "Ветровые данные не загружены",
                "Вы добавили ВЭУ, но не загрузили данные о ветре. "
                "Продолжить оптимизацию без учета реальных ветровых условий?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return False

        return True

    # -------------------------------------------------------------------------
    # ОБРАБОТКА ЗАВЕРШЕНИЯ/ОШИБОК ОПТИМИЗАЦИИ
    # -------------------------------------------------------------------------
    def on_optimization_finished(self, result):
        """
        Вызывается в главном потоке, когда OptimizationThread.finished.emit(result) сработал.

        Действия:
        1) сохраняем результат;
        2) возвращаем UI в "готовое" состояние (кнопка активна, progress=100%);
        3) проверяем результат (validate_optimization_results);
        4) обновляем таблицы и визуализацию.
        """
        self.optimization_result = result

        # Разблокируем кнопку
        self.main_window.optimize_btn.setEnabled(True)

        # Показываем финал прогресса (красиво).
        self.main_window.progress_bar.setRange(0, 100)
        self.main_window.progress_bar.setValue(100)
        self.main_window.progress_bar.setFormat("100% — готово")

        # Валидация (мягкие предупреждения пользователю)
        self.validate_optimization_results(result)

        # Пишем таблицы результатов
        self.main_window.update_results_tables(result)

        # Обновляем диапазон дат (если schedule/index содержит реальные даты)
        self.update_date_selector_range()

        # Перерисовываем графики под текущую выбранную дату
        self.update_visualization()

        QMessageBox.information(self.main_window, "Успех", "Оптимизация завершена успешно")

    def validate_optimization_results(self, result):
        """
        Проверка результата "на здравый смысл".

        Делает:
        • проверку непокрытой нагрузки (unserved);
        • оценку доли ВИЭ (renewable_to_load);
        • логирует одновременный заряд/разряд АКБ (по первым 24 часам).

        ⚠️ Важно:
        Это НЕ строгая проверка корректности алгоритма,
        а именно пользовательские предупреждения, чтобы не получить "тихий" мусор.
        """
        schedule = result.schedule

        # 1) Непокрытая нагрузка (если >0 — решение физически/структурно плохое)
        total_unserved = schedule['unserved'].sum()
        if total_unserved > 0.1:
            QMessageBox.warning(
                self.main_window, "Предупреждение",
                f"Обнаружена непокрытая нагрузка: {total_unserved:.2f} кВт·ч\n"
                f"Рекомендуется увеличить мощность состава генерации или расширить каталог оборудования"
            )

        # 2) Доля ВИЭ по энергии, покрывшей нагрузку
        total_load = schedule['load'].sum()
        total_renewable = schedule['renewable_to_load'].sum()
        renewable_share = (total_renewable / total_load * 100) if total_load > 0 else 0

        if renewable_share < 20:
            QMessageBox.information(
                self.main_window, "Информация",
                f"Доля ВИЭ составляет {renewable_share:.1f}%\n"
                f"Рекомендуется рассмотреть увеличение мощности ВИЭ"
            )

        # 3) Одновременный заряд/разряд АКБ — индикатор ошибки диспетчеризации/баланса
        for hour in range(min(24, len(schedule))):
            charge = (
                schedule.iloc[hour].get('battery_charge_from_renewable', 0) +
                schedule.iloc[hour].get('battery_charge_from_diesel', 0)
            )
            discharge = schedule.iloc[hour].get('battery_discharge', 0)

            if charge > 0.1 and discharge > 0.1:
                print(f"ВНИМАНИЕ: Одновременный заряд и разряд АКБ на час {hour + 1}")

    def on_optimization_error(self, error_message: str):
        """
        Обработка ошибки оптимизации.

        Вызывается в главном потоке (через signal).
        Возвращает UI в норму и показывает окно ошибки.
        """
        self.main_window.optimize_btn.setEnabled(True)
        self.main_window.progress_bar.setVisible(False)
        QMessageBox.critical(self.main_window, "Ошибка", f"Ошибка оптимизации: {error_message}")

    # -------------------------------------------------------------------------
    # ВИЗУАЛИЗАЦИЯ
    # -------------------------------------------------------------------------
    def update_visualization(self):
        """
        Публичный "триггер" перерисовки графиков/таблиц под выбранную дату.

        Вызывается:
        • при смене date_selector;
        • после завершения оптимизации;
        • потенциально после загрузки данных.

        Реальная логика сидит в _update_visualization_by_date().
        """
        if self.optimization_result:
            try:
                self._update_visualization_by_date()
            except Exception as e:
                print(f"Ошибка визуализации: {e}")
                # В случае проблем рисуем заглушку вместо падения UI
                try:
                    self.main_window.visualization_widget.draw_no_data_message()
                except Exception:
                    pass

    def get_available_date_range(self):
        """
        Определяет доступный диапазон дат для date_selector.

        Источники (по приоритету):
        1) optimization_result.schedule.index, если это DatetimeIndex.
        2) load_profile.data.index, если это DatetimeIndex.
        3) иначе возвращаем (None, None) -> будет fallback 2021.

        Возвращает:
        • (min_date, max_date) как datetime.date, либо (None, None).
        """
        schedule_index = None
        if self.optimization_result and hasattr(self.optimization_result, 'schedule'):
            schedule_index = getattr(self.optimization_result.schedule, 'index', None)

        if isinstance(schedule_index, pd.DatetimeIndex):
            return schedule_index.min().date(), schedule_index.max().date()

        if self.load_profile and hasattr(self.load_profile, 'data'):
            idx = getattr(self.load_profile.data, 'index', None)
            if isinstance(idx, pd.DatetimeIndex):
                return idx.min().date(), idx.max().date()

        return None, None

    def update_date_selector_range(self):
        """
        Выставляет диапазон доступных дат в UI.

        Если есть реальные даты из данных — используем их.
        Если нет — fallback 2021-01-01..2021-12-31 (как в MainWindow).
        """
        min_date, max_date = self.get_available_date_range()
        if min_date and max_date:
            self.main_window.set_date_range(
                QDate(min_date.year, min_date.month, min_date.day),
                QDate(max_date.year, max_date.month, max_date.day)
            )
        else:
            fallback_start = QDate(2021, 1, 1)
            fallback_end = QDate(2021, 12, 31)
            self.main_window.set_date_range(fallback_start, fallback_end)

    def _update_visualization_by_date(self):
        """
        Перерисовать EnergyChart и BatteryChart для выбранной даты.

        Входные данные:
        • self.optimization_result.schedule — DataFrame
          обычно с длиной 8760 (или больше) и колонками:
          load, hydro, wind, diesel, dump, unserved, soc, ... etc.

        Алгоритм определения start_idx (откуда брать 24 часа):
        1) если schedule.index — DatetimeIndex:
           находим все строки выбранной даты (normalize) и берем первый час;
        2) иначе считаем, что это "годовой массив" где день -> блок из 24 рядов:
           day_of_year -> start_idx = (day-1)*24.

        Затем:
        • дергаем EnergyChart.plot_day(schedule, start_idx, title);
        • достаем day_slice = schedule.iloc[start_idx:start_idx+24];
        • строим battery_meta (агрегируем параметры батарей из результата);
        • считаем SOC на начало суток:
          — если есть пред.час (start_idx>0) и есть колонка 'soc', берем его;
          — иначе берем meta['soc_start_kwh'], если есть.
        • рисуем BatteryChart.plot_battery_day(...) и обновляем таблицу summary.
        """
        schedule = self.optimization_result.schedule

        selected_qdate = self.main_window.get_selected_date()
        if not selected_qdate.isValid():
            return

        # QDate -> datetime.date
        selected_date = selected_qdate.toPython()

        # Красивый заголовок (но зависит от корректности format_ru_date в MainWindow)
        formatted_date = self.main_window.format_ru_date(selected_date)

        # 1) если индекс содержит реальные даты
        if isinstance(schedule.index, pd.DatetimeIndex):
            normalized = schedule.index.normalize()
            matches = normalized == pd.Timestamp(selected_date)
            if not matches.any():
                QMessageBox.warning(self.main_window, "Нет данных",
                                    "Для выбранной даты нет данных в графике нагрузки")
                return
            start_idx = int(np.flatnonzero(matches)[0])
        else:
            # 2) fallback: день года -> блок из 24 часов
            day_of_year = selected_qdate.dayOfYear()
            start_idx = (day_of_year - 1) * 24

        # Блок проверок границ (пользователь может выбрать дату вне данных)
        if start_idx >= len(schedule):
            QMessageBox.warning(self.main_window, "Нет данных",
                                "Выбранная дата выходит за пределы доступных данных")
            return

        if start_idx + 24 > len(schedule):
            QMessageBox.warning(self.main_window, "Нет данных",
                                "Для выбранной даты недостаточно данных для построения суток")
            return

        # ---- EnergyChart ----
        self.main_window.update_chart_header(formatted_date)
        self.main_window.visualization_widget.plot_day(
            schedule,
            start_idx,
            f"Покрытие нагрузки ({formatted_date})"
        )

        # ---- BatteryChart ----
        day_slice = schedule.iloc[start_idx:start_idx + 24]
        battery_meta = self._build_battery_meta()

        self.main_window.update_battery_header(formatted_date)

        # SOC на начало суток: берем carry-over если доступно, иначе initial.
        day_start_soc = None
        start_source = ""

        # ⚠️ Тут subtle bug: ты берешь schedule.iloc[start_idx - 1]['soc'] (скорее SOC в кВт*ч),
        # но в BatteryChart логика ожидает day_start_soc как kWh; это ок.
        # Однако если 'soc' у тебя хранится в долях (0..1), то будет неверно.
        if start_idx > 0 and 'soc' in schedule.columns:
            day_start_soc = float(schedule.iloc[start_idx - 1]['soc'])
            start_source = "carry"
        elif battery_meta.get('soc_start_kwh') is not None:
            day_start_soc = float(battery_meta['soc_start_kwh'])
            start_source = "initial"

        try:
            profile = self.main_window.battery_chart.plot_battery_day(
                day_slice,
                formatted_date,
                battery_meta,
                day_start_soc=day_start_soc,
                soc_start_source=start_source,
            )
            self.main_window.update_battery_table(profile)

        except ValueError as exc:
            QMessageBox.warning(self.main_window, "Ошибка визуализации АКБ", str(exc))
            try:
                self.main_window.battery_chart.draw_no_data_message(str(exc))
                self.main_window.update_battery_table({'warnings': [str(exc)]})
            except Exception:
                pass

    def _build_battery_meta(self) -> dict:
        """
        Собирает метаданные батареи для BatteryChart на основе:
        • optimization_result.selected_equipment['batteries'] (dict-ы);
        • self.batteries (каталог объектов Battery), чтобы подтянуть недостающие поля.

        Почему это нужно:
        • в schedule мы видим только поток энергии (заряд/разряд/SOC);
        • для корректной проверки SOC и графика нужны пределы soc_min/soc_max и КПД.

        Что делает:
        1) берет батареи из результата (выбранные), сопоставляет их по name с объектами каталога;
        2) агрегирует:
           — суммарную емкость capacity_total;
           — минимальный soc_min (самый жесткий нижний предел);
           — максимальный soc_max (самый жесткий верхний предел);
           — при наличии soc_min_kwh/soc_max_kwh — берет min/max по kWh;
           — КПД усредняет по всем найденным значениям.
        3) soc_start_kwh/soc_end_kwh берет из result_batteries, если там есть.

        ⚠️ Важно про агрегирование soc_min/soc_max:
        — сейчас берется min(soc_min) и max(soc_max), это "расширяет" допустимую зону,
          если батареи разные. Для строгой безопасности чаще нужно:
          — soc_min = max(soc_mins) (наиболее высокий минимум),
          — soc_max = min(soc_maxs) (наиболее низкий максимум),
          чтобы не нарушить ограничения ни одной батареи.
        """
        meta = {
            'capacity_total': 0,
            'soc_min': None,
            'soc_max': None,
            'soc_min_kwh': None,
            'soc_max_kwh': None,
            'eta_charge': None,
            'eta_discharge': None,
            'soc_start_kwh': None,
            'soc_end_kwh': None,
            'soc_start_source': '',
        }

        result_batteries = []
        if self.optimization_result and hasattr(self.optimization_result, 'selected_equipment'):
            result_batteries = self.optimization_result.selected_equipment.get('batteries', [])

        capacities = []
        soc_mins = []
        soc_maxs = []
        soc_mins_kwh = []
        soc_maxs_kwh = []
        etas = []

        for bat in result_batteries:
            name = bat.get('name')
            capacity = bat.get('capacity', 0)

            # Сопоставляем с объектами Battery из каталога
            matched = next((b for b in self.batteries if getattr(b, 'name', None) == name), None)

            # soc_min/soc_max могут быть в result или в каталоге (matched)
            soc_min = bat.get('soc_min', getattr(matched, 'soc_min', None))
            soc_max = bat.get('soc_max', getattr(matched, 'soc_max', None))

            # в kWh иногда уже посчитано заранее
            soc_min_kwh = bat.get('soc_min_kwh')
            soc_max_kwh = bat.get('soc_max_kwh')

            # КПД: разные схемы обозначения (eta_charge/efficiency)
            eta = bat.get('eta_charge', bat.get('efficiency', getattr(matched, 'efficiency', None)))
            eta_dis = bat.get('eta_discharge', eta)

            capacities.append(capacity)
            if soc_min is not None:
                soc_mins.append(soc_min)
            if soc_max is not None:
                soc_maxs.append(soc_max)
            if soc_min_kwh is not None:
                soc_mins_kwh.append(soc_min_kwh)
            if soc_max_kwh is not None:
                soc_maxs_kwh.append(soc_max_kwh)
            if eta is not None:
                etas.append(eta)
            if eta_dis is not None and eta_dis != eta:
                etas.append(eta_dis)

            # Начальный/конечный SOC (если оптимизатор их сохранил)
            if meta['soc_start_kwh'] is None and bat.get('soc_start_kwh') is not None:
                meta['soc_start_kwh'] = float(bat['soc_start_kwh'])
                meta['soc_start_source'] = 'initial'
            if meta['soc_end_kwh'] is None and bat.get('soc_end_kwh') is not None:
                meta['soc_end_kwh'] = float(bat['soc_end_kwh'])

        # Агрегирование
        if capacities:
            meta['capacity_total'] = float(sum(capacities))
        if soc_mins:
            meta['soc_min'] = float(min(soc_mins))   # см. заметку выше про строгость
        if soc_maxs:
            meta['soc_max'] = float(max(soc_maxs))   # см. заметку выше про строгость
        if soc_mins_kwh:
            meta['soc_min_kwh'] = float(min(soc_mins_kwh))
        if soc_maxs_kwh:
            meta['soc_max_kwh'] = float(max(soc_maxs_kwh))
        if etas:
            eta_avg = float(sum(etas) / len(etas))
            meta['eta_charge'] = eta_avg
            meta['eta_discharge'] = eta_avg

        return meta

    def run(self):
        """
        Запуск UI и main loop Qt.

        Важно:
        • show() должен быть вызван ДО exec(), иначе окно не появится.
        • app.exec() блокирует поток до закрытия приложения.
        """
        self.main_window.show()
        return self.app.exec()


if __name__ == "__main__":
    """
    Entry point.

    sys.path.append(...) нужен для корректного импорта модулей,
    если запускаешь main.py напрямую, а не как пакет.

    Затем создается EnergyOptimizerApp и запускается цикл событий.
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    optimizer = EnergyOptimizerApp()
    sys.exit(optimizer.run())
