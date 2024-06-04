import os

from fpdf import FPDF
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from numpy import arange, around, tile, linspace
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from config.config import config_provider_instance
from config.strings import get_string, Strings
from ui.main_view import MainView, InputValues, OutputValues, CellColor
from utils.utils import get_mpl_color


class MainPresenter:
    _view: MainView
    _output_values: OutputValues

    def __init__(self, view: MainView):
        self._view = view

    def on_view_created(self):
        values = InputValues()
        config = config_provider_instance.get_config()
        values.mass = config.mass
        values.frequency_from = config.frequency_from
        values.frequency_to = config.frequency_to
        values.frequency_step = config.frequency_step
        values.energy_from = config.energy_from
        values.energy_to = config.energy_to
        values.energy_step = config.energy_step
        values.gamma = config.gamma
        self._view.set_initial_values(values)
        self._output_values = self._calculate(values)
        self._view.show_result(self._output_values)

    def on_exit_clicked(self):
        self._view.exit()

    def on_calculate_clicked(self, values: InputValues):
        self._view.show_progress(True)
        try:
            self._output_values = self._calculate(values)
            self._view.show_result(self._output_values)
            self._view.show_safe_button(True)
        except:
            self._view.show_safe_button(False)
            self._view.show_error_dialog(get_string(Strings.MESSAGE_ERROR_CALCULATION))

        self._view.show_progress(False)

    def _calculate(self, values: InputValues):
        config = config_provider_instance.get_config()
        round_numbers = config.round_numbers_count

        energy_from = float(values.energy_from)
        energy_to = float(values.energy_to)
        energy_step = float(values.energy_step)
        frequency_from = float(values.frequency_from)
        frequency_to = float(values.frequency_to)
        frequency_step = float(values.frequency_step)
        mass = float(values.mass)
        gamma = float(values.gamma)

        energy_range = arange(energy_from, energy_to + energy_step, energy_step)
        frequency_range = arange(frequency_from, frequency_to + frequency_step, frequency_step)
        energy = tile(energy_range.reshape(-1, 1), (1, frequency_range.size))
        frequency = tile(frequency_range.reshape(1, -1), (energy_range.size, 1))

        result = around(mass / (gamma * energy * frequency), round_numbers)

        bounds = linspace(result.min(), result.max(), len(config.color_theme) + 1)
        cell_colors = []
        for i in range(len(bounds) - 1):
            color = CellColor()
            color.min = bounds[i]
            color.max = bounds[i + 1]
            color.background = config.color_theme[i]
            color.text = config.color_text_theme[i]
            cell_colors.append(color)

        figure = Figure()
        ax = figure.add_subplot(121, projection="3d")
        ax.set_xlabel("Энергия, Дж")
        ax.set_ylabel("Частота, Гц")
        # ax.set_zlabel("Время, сек.")
        st_angle = 20
        angle = -25
        ax.view_init(st_angle, angle)
        ax.plot_surface(energy, frequency, result,
                        cmap=ListedColormap(list(map(lambda c: get_mpl_color(c), config.color_theme))))

        ax = figure.add_subplot(122)
        img = plt.imread('./icons/Recomend.png')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(img)

        frequency_labels = [str(round(i, round_numbers)) for i in frequency[0]]
        energy_labels = [str(round(i[0], round_numbers)) for i in energy]

        res = OutputValues()
        res.frequency_labels = frequency_labels
        res.energy_labels = energy_labels
        res.values = result
        res.figure = figure
        res.cell_colors = cell_colors

        return res

    def on_save_clicked(self):
        self._view.show_progress(True)
        self._view.show_select_directory_dialog()

    def on_output_path_selected(self, path):
        self._view.show_progress(False)
        if len(path) == 0:
            return

        temp_image_name = ".saved_figure.png"
        output = self._output_values
        output.figure.savefig(temp_image_name, transparent=True, bbox_inches='tight', format='png')
        pdf = FPDF()
        pdf.add_font('DejaVu', '', r"\SUD\fonts\DejaVuSerifCondensed.ttf", uni=True)
        pdf.set_font('DejaVu', '', 14)

        # pdf.set_font("Times", size=11)
        pdf.add_page()
        width = 150
        height = 5
        col_width = pdf.w / 3.5
        row_height = pdf.font_size
        spacing = 1

        # pdf.cell(width, height, txt=f'Масса камня: {self.Inpu} гр.', ln=1, align="L")
        # pdf.add_page()
        pdf.cell(30, height, "Частота, Гц", 1, align="C")
        for i in output.frequency_labels:
            pdf.cell(10, height * 2, i, 1, align="C")
        pdf.ln(height)
        pdf.cell(30, height, "Энергия, Дж", 1, align="C")
        pdf.ln(height)

        for i in range(len(output.values)):
            pdf.cell(30, height, output.energy_labels[i], 1, fill=False, align="C")
            for j in range(len(output.values[i])):
                for color in output.cell_colors:
                    if color.min <= output.values[i][j] <= color.max:
                        pdf.set_fill_color(*color.background)
                        pdf.set_text_color(*color.text)
                        break
                pdf.cell(8, height, str(output.values[i][j]), 1, fill=True)
            pdf.ln(height)
            pdf.set_text_color(0, 0, 0)
        x_pos = pdf.get_x()
        y_pos = pdf.get_y()

        pdf.image(".saved_figure.png", x=x_pos + 107, y=y_pos - 100, w=90, h=70)

        recommended = [
            ['Режим для дробления осколков камней', ''],
            ['Режим работы для камней с низкой плотностью ', ''],
            ['Режим работы при неудобных подходах инструмента', ''],
            ['Оптимальный режим работы', ''],
            ['Режим работы в зоне высокой плотности камня', ''],
            ['Режим работы с высоким риском нанесения травмы', ''],
            ['Малоэффективный режим работы', ''],
            ['Малоэффективный режим работы – не рекомендовано к применению', '']
        ]
        rec_colors = [
            (57, 73, 171),
            (3, 155, 229),
            (0, 172, 193),
            (0, 137, 123),
            (67, 160, 71),
            (124, 179, 66),
            (142, 36, 170),
            (216, 27, 96)
        ]
        pdf.ln(row_height * spacing)
        i = 0
        pdf.cell(14, row_height * spacing, txt='Рекомендации по подбору режима работы лазера')
        pdf.ln(row_height * spacing)
        for row in recommended:
            # for item in row:
            pdf.set_fill_color(*rec_colors[i])
            pdf.cell(100, row_height * spacing, txt=row[0], border=1, fill=False)
            pdf.cell(20, row_height * spacing, txt=row[1], border=1, fill=True)
            pdf.ln(row_height * spacing)
            i += 1
        pdf.ln(row_height * spacing)

        pdf.output(path + ".pdf")

        try:
            os.remove(temp_image_name)
        except FileNotFoundError:
            pass
