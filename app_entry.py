# File: app_entry.py
import velopack

def main() -> int:
    # Должно быть самым первым, до старта Qt и тяжелых импортов.
    # В некоторых случаях Velopack может завершить/перезапустить процесс.
    velopack.App().run()

    # Теперь безопасно импортировать все остальное
    from main import EnergyOptimizerApp  # main.py не должен запускаться при импорте

    app = EnergyOptimizerApp()
    return app.run()

if __name__ == "__main__":
    raise SystemExit(main())
