import sys
import argparse

from PyQt5.QtWidgets import QApplication

from config.config import Config, config_provider_instance
from ui.main_view_impl import MainViewImpl


def start(config: Config):
    config_provider_instance.init(config)
    app = QApplication(sys.argv)
    win = MainViewImpl()
    win.show()
    app.exec_()

    win.close()
    app.exit()


def main_laser(stone_mass):
    start(Config.create_default(stone_mass=stone_mass))


if __name__ == '__main__':
    stone_mass = 0.984
    start(Config.create_default(stone_mass=stone_mass))
