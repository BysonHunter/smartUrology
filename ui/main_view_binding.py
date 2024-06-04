from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QLineEdit, QProgressBar, \
    QTableWidget, QAbstractItemView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from config.strings import get_string, Strings


class InputWithLabel(QWidget):
    line: QLineEdit

    def __init__(self, label):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.line = QLineEdit()
        layout.addWidget(QLabel(label))
        layout.addWidget(self.line)


class InputWithLabelAndSteps(QWidget):
    fromLine: InputWithLabel
    toLine: InputWithLabel
    stepLine: InputWithLabel

    def __init__(self, label):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)
        input_layout = QHBoxLayout()

        self.fromLine = InputWithLabel(get_string(Strings.LABEL_FROM))
        self.toLine = InputWithLabel(get_string(Strings.LABEL_TO))
        self.stepLine = InputWithLabel(get_string(Strings.LABEL_STEP))

        layout.addWidget(QLabel(label))
        layout.addLayout(input_layout)
        input_layout.addWidget(self.fromLine)
        input_layout.addWidget(self.toLine)
        input_layout.addWidget(self.stepLine)


class MainViewBinding:
    root: QWidget
    gammaInput: InputWithLabel
    massInput: InputWithLabel
    energyInput: InputWithLabelAndSteps
    frequencyInput: InputWithLabelAndSteps
    progressBar: QProgressBar
    calculateButton: QPushButton
    buttonWidget: QWidget
    saveButton: QPushButton
    exitButton: QPushButton
    table: QTableWidget
    canvas: FigureCanvas

    @staticmethod
    # @classmethod
    def inflate():
        root = QWidget()
        root_layout = QHBoxLayout()
        root.setLayout(root_layout)

        control_layout = QVBoxLayout()
        control_layout.setAlignment(Qt.AlignTop)
        root_layout.addLayout(control_layout, 1)

        gamma_input_widget = InputWithLabel(get_string(Strings.INPUT_GAMMA))
        control_layout.addWidget(gamma_input_widget)

        mass_input_widget = InputWithLabel(get_string(Strings.INPUT_MASS))
        control_layout.addWidget(mass_input_widget)

        frequency_input_widget = InputWithLabelAndSteps(get_string(Strings.INPUT_FREQUENCY))
        control_layout.addWidget(frequency_input_widget)

        energy_input_widget = InputWithLabelAndSteps(get_string(Strings.INPUT_ENERGY))
        control_layout.addWidget(energy_input_widget)

        progress_bar = QProgressBar()
        progress_bar.hide()
        progress_bar.setMaximum(0)
        progress_bar.setMinimum(0)
        control_layout.addWidget(progress_bar)

        buttons_layout = QVBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        calculate_button = QPushButton(get_string(Strings.BUTTON_CALCULATE))
        buttons_layout.addWidget(calculate_button)

        save_button = QPushButton(get_string(Strings.BUTTON_SAVE))
        save_button.hide()
        buttons_layout.addWidget(save_button)

        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons_layout)

        control_layout.addWidget(buttons_widget)

        exit_button = QPushButton(get_string(Strings.BUTTON_EXIT))
        control_layout.addWidget(exit_button)

        graph_layout = QVBoxLayout()
        root_layout.addLayout(graph_layout, 14)

        table = QTableWidget()
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        graph_layout.addWidget(table, 1)

        canvas = FigureCanvas()
        graph_layout.addWidget(canvas, 1)

        binding = MainViewBinding()
        binding.root = root
        binding.gammaInput = gamma_input_widget
        binding.massInput = mass_input_widget
        binding.energyInput = energy_input_widget
        binding.frequencyInput = frequency_input_widget
        binding.progressBar = progress_bar
        binding.calculateButton = calculate_button
        binding.buttonWidget = buttons_widget
        binding.saveButton = save_button
        binding.exitButton = exit_button
        binding.table = table
        binding.canvas = canvas

        return binding
