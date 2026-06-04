from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger
from PySide6.QtWidgets import QApplication


def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)


def _setup_qt_plugin_path() -> None:
    pyside6_dir = Path(sys.prefix) / "Lib" / "site-packages" / "PySide6"
    plugin_dir = pyside6_dir / "plugins"
    platform_dir = plugin_dir / "platforms"
    if plugin_dir.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_dir))
    if platform_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platform_dir))


def _setup_loguru_file_sink() -> None:
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "test_gui_service.log",
        level="DEBUG",
        encoding="utf-8",
        rotation="5 MB",
        retention=5,
    )
    logger.info("Test GUI log file initialized: {}", log_dir / "test_gui_service.log")


def main() -> int:
    _ensure_project_root_on_path()
    _setup_qt_plugin_path()
    _setup_loguru_file_sink()

    from gui.test_gui.test_main_view import TestMainView

    app = QApplication.instance() or QApplication(sys.argv)
    window = TestMainView()
    window.resize(1388, 894)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
