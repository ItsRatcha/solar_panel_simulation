"""Entry point: python -m pvsim.gui"""

from __future__ import annotations

import sys


def main() -> int:
    import pyqtgraph as pg
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    pg.setConfigOptions(antialias=True)
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
