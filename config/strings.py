from enum import Enum

from config.config import config_provider_instance


class Strings(Enum):
    TITLE_WINDOW = 1
    TITLE_SAFE_FILE_DIALOG = 2
    TITLE_ERROR_DIALOG = 3

    BUTTON_CALCULATE = 100 + 1
    BUTTON_EXIT = 100 + 2
    BUTTON_SAVE = 100 + 3

    INPUT_GAMMA = 10000 + 1
    INPUT_ENERGY = 10000 + 2
    INPUT_FREQUENCY = 10000 + 3
    INPUT_MASS = 10000 + 4

    LABEL_FROM = 1000000 + 1
    LABEL_TO = 1000000 + 2
    LABEL_STEP = 1000000 + 3

    MESSAGE_ERROR_CALCULATION = 100000000 + 1


def get_string(string: Strings):
    locale = config_provider_instance.get_config().locale
    if string == Strings.TITLE_WINDOW:
        if locale == "RU":
            return "Расчет времени разрушения камня"
    elif string == Strings.TITLE_SAFE_FILE_DIALOG:
        if locale == "RU":
            return "Выберите файл"
    elif string == Strings.TITLE_ERROR_DIALOG:
        if locale == "RU":
            return "Ошибка"
    elif string == Strings.BUTTON_CALCULATE:
        if locale == "RU":
            return "Рассчитать"
    elif string == Strings.BUTTON_EXIT:
        if locale == "RU":
            return "Выход"
    elif string == Strings.BUTTON_SAVE:
        if locale == "RU":
            return "Сохранить"
    elif string == Strings.INPUT_GAMMA:
        if locale == "RU":
            return "гамма-излучение"
    elif string == Strings.INPUT_ENERGY:
        if locale == "RU":
            return "энергия лазера"
    elif string == Strings.INPUT_FREQUENCY:
        if locale == "RU":
            return "частота излучения лазера"
    elif string == Strings.INPUT_MASS:
        if locale == "RU":
            return "масса камня"
    elif string == Strings.LABEL_FROM:
        if locale == "RU":
            return "начальное значение"
    elif string == Strings.LABEL_TO:
        if locale == "RU":
            return "конечное значение"
    elif string == Strings.LABEL_STEP:
        if locale == "RU":
            return "шаг"
    elif string == Strings.MESSAGE_ERROR_CALCULATION:
        if locale == "RU":
            return "Введены некорректные значения"
