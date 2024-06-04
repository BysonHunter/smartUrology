from pathlib import Path

from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem, QFileDialog, QErrorMessage, QMessageBox

from config.strings import Strings, get_string
from ui.main_presenter import MainPresenter
from ui.main_view import MainView, InputValues, OutputValues
from ui.main_view_binding import MainViewBinding


class MainViewImpl(QMainWindow, MainView):
    _presenter: MainPresenter
    _binding: MainViewBinding

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Рассчет времени разрушения камня")
        self._init_presenter()
        self._init_ui()
        self._init_listeners()
        self._presenter.on_view_created()

    def show_progress(self, is_visible: bool):
        if is_visible:
            self._binding.progressBar.show()
            self._binding.buttonWidget.hide()
        else:
            self._binding.progressBar.hide()
            self._binding.buttonWidget.show()

    def show_safe_button(self, is_visible: bool):
        if is_visible:
            self._binding.saveButton.show()
        else:
            self._binding.saveButton.hide()

    def set_initial_values(self, values: InputValues):
        self._binding.gammaInput.line.setText(str(values.gamma))
        self._binding.energyInput.fromLine.line.setText(str(values.energy_from))
        self._binding.massInput.line.setText(str(values.mass))
        self._binding.energyInput.toLine.line.setText(str(values.energy_to))
        self._binding.energyInput.stepLine.line.setText(str(values.energy_step))
        self._binding.frequencyInput.fromLine.line.setText(str(values.frequency_from))
        self._binding.frequencyInput.toLine.line.setText(str(values.frequency_to))
        self._binding.frequencyInput.stepLine.line.setText(str(values.frequency_step))

    def show_result(self, values: OutputValues):
        table = self._binding.table
        #table1 = self._binding.table

        while table.rowCount() > 0:
            table.removeRow(0)

        while table.columnCount() > 0:
            table.removeColumn(0)

        for i in range(len(values.energy_labels)):
            table.insertRow(i)

        for i in range(len(values.frequency_labels)):
            table.insertColumn(i)

        for i in range(len(values.values)):
            for j in range(len(values.values[i])):
                value = values.values[i][j]
                item = QTableWidgetItem(str(value))
                for color in values.cell_colors:
                    if color.min <= value <= color.max:
                        item.setBackground(QBrush(QColor(*color.background)))
                        item.setForeground(QBrush(QColor(*color.text)))
                        break
                table.setItem(i, j, item)

        table.setHorizontalHeaderLabels(values.frequency_labels)
        table.setVerticalHeaderLabels(values.energy_labels)

        canvas = self._binding.canvas
        canvas.figure = values.figure
        canvas.draw()

    def show_select_directory_dialog(self):
        path = QFileDialog.getSaveFileName(self, get_string(Strings.TITLE_SAFE_FILE_DIALOG), str(Path.home()))
        self._presenter.on_output_path_selected(path[0])

    def show_error_dialog(self, message):
        msg = QMessageBox(self)
        msg.setText(message)
        msg.setWindowTitle(get_string(Strings.TITLE_ERROR_DIALOG))
        msg.exec()

    def exit(self):
        self.window().close()

    def _init_presenter(self):
        self._presenter = MainPresenter(self)

    def _init_ui(self):
        self._binding = MainViewBinding.inflate()
        self.setWindowTitle(get_string(Strings.TITLE_WINDOW))
        self.setCentralWidget(self._binding.root)

    def _init_listeners(self):
        self._binding.exitButton.clicked.connect(self._presenter.on_exit_clicked)
        self._binding.calculateButton.clicked.connect(self._on_calculate_clicked)
        self._binding.saveButton.clicked.connect(self._presenter.on_save_clicked)

    def _on_calculate_clicked(self):
        values = InputValues()
        values.mass = self._binding.massInput.line.text()
        values.gamma = self._binding.gammaInput.line.text()
        values.energy_from = self._binding.energyInput.fromLine.line.text()
        values.energy_to = self._binding.energyInput.toLine.line.text()
        values.energy_step = self._binding.energyInput.stepLine.line.text()
        values.frequency_from = self._binding.frequencyInput.fromLine.line.text()
        values.frequency_to = self._binding.frequencyInput.toLine.line.text()
        values.frequency_step = self._binding.frequencyInput.stepLine.line.text()
        self._presenter.on_calculate_clicked(values)
