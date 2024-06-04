from matplotlib.figure import Figure


class InputValues:
    gamma: str

    frequency_from: str
    frequency_to: str
    frequency_step: str

    energy_from: str
    energy_to: str
    energy_step: str

    mass: str


class CellColor:
    min: float
    max: float
    background: str
    text: str


class OutputValues:
    frequency_labels: [str]
    energy_labels: [str]
    values: [[float]]
    figure: Figure
    cell_colors: [CellColor]


class MainView:
    def show_progress(self, is_visible: bool):
        pass

    def show_safe_button(self, is_visible: bool):
        pass

    def set_initial_values(self, values: InputValues):
        pass

    def show_result(self, values: OutputValues):
        pass

    def show_select_directory_dialog(self):
        pass

    def show_error_dialog(self, message):
        pass

    def exit(self):
        pass
