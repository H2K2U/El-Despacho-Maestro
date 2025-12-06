# File: core/updater.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import velopack


@dataclass
class UpdateResult:
    """Результат проверки обновлений."""
    has_update: bool
    message: str = ""
    # update_info нельзя типизировать жестко без знания внутреннего класса velopack,
    # но его нужно передавать в download/apply.
    update_info: Optional[object] = None


def check_for_updates(feed_url: str) -> UpdateResult:
    """
    Проверяет наличие обновлений в "feed_url".

    feed_url — базовый URL, где лежат файлы обновлений Velopack (metadata + пакеты).
    Пример: "https://your-domain/updates"
    """
    manager = velopack.UpdateManager(feed_url)

    update_info = manager.check_for_updates()
    if not update_info:
        return UpdateResult(False, "Нет доступных обновлений")

    return UpdateResult(True, "Доступно обновление", update_info)


def download_and_apply_update(feed_url: str, update_info: object) -> None:
    """
    Скачивает обновление и применяет его с перезапуском приложения.

    Важно: apply_updates_and_restart завершит текущий процесс и запустит обновленную версию.
    """
    manager = velopack.UpdateManager(feed_url)
    manager.download_updates(update_info)
    manager.apply_updates_and_restart(update_info)
