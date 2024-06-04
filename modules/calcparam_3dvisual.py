import csv
from os import listdir
from pathlib import Path
from sys import maxsize as max_int_size
import PySimpleGUI as sg
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import plotly
import os, sys
import plotly.graph_objs as go
import PyQt5
from PyQt5 import QtWebEngineWidgets
from PyQt5.QtWidgets import QDesktopWidget, QApplication
from PyQt5.QtCore import QUrl
import matplotlib
import matplotlib.colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules import config as cf
from modules import pdf_working as pw
from modules import getimagefromdicom_new as gi


min_int_size = -max_int_size - 1
left_kidney = 0
left_kidney_pieloectasy = 3
right_kidney = 2
right_kidney_pieloectasy = 4
stone = 1
staghorn_stones = 5
overlaps_min_light = 160
left_kidney_center_constaints_x = (0.55, 0.704613)
left_kidney_center_constaints_z = (0.248879, 0.66704)
right_kidney_center_constaints_x = (0.2, 0.45)
right_kidney_center_constaints_z = (0.2, 0.804613)

class SliceDTO:
    def __init__(self, label, x, z, w, h, conf):
        self.label = label
        self.x = x
        self.z = z
        self.w = w
        self.h = h
        self.conf = conf
        
        self.min_x = self.x - w / 2
        self.max_x = self.x + w / 2
        self.min_z = self.z - h / 2
        self.max_z = self.z + h / 2
        
        
class LayerDTO:
    def __init__(self, y, slice_list):
        self.y = y
        self.slice_list = slice_list
        
        
class ObjectParamsDto:
    def __init__(self, x_center, z_center, w, h, number, first_index, last_index):
        self.x_center = x_center
        self.z_center = z_center
        self.w = w
        self.h = h
        self.number = number
        self.first_index = first_index
        self.last_index = last_index


def is_in_other_slice(cur, other):
    is_in_x = other.min_x < cur.x < other.max_x
    is_in_z = other.min_z < cur.z < other.max_z
    return is_in_x and is_in_z


def is_in_left_kidney_constraints(slice):
    is_x_in_constraints = left_kidney_center_constaints_x[0] < slice.x < left_kidney_center_constaints_x[1]
    is_z_in_constraints = left_kidney_center_constaints_z[0] < slice.z < left_kidney_center_constaints_z[1]
    return is_x_in_constraints and is_z_in_constraints


def is_in_right_kidney_constraints(slice):
    is_x_in_constraints = right_kidney_center_constaints_x[0] < slice.x < right_kidney_center_constaints_x[1]
    is_z_in_constraints = right_kidney_center_constaints_z[0] < slice.z < right_kidney_center_constaints_z[1]
    return is_x_in_constraints and is_z_in_constraints


def is_stone(slice):
    return slice.label == staghorn_stones or slice.label == stone


def is_right_kidney(slice):
    return slice.label == right_kidney or slice.label == right_kidney_pieloectasy


def is_left_kidney(slice):
    return slice.label == left_kidney or slice.label == left_kidney_pieloectasy


def get_array_indexes(x_beg, x_end, z_beg, z_end, shape):
    return int(shape[2]*(x_beg)), int(shape[2]*(x_end)), int(shape[0]*(z_beg)), int(shape[0]*(z_end))


def get_array_indexes_2d(x_beg, x_end, z_beg, z_end, shape):
    return int(shape[1]*(x_beg)), int(shape[1]*(x_end)), int(shape[0]*(z_beg)), int(shape[0]*(z_end))


def get_subarray_2d(x_beg, x_end, z_beg, z_end, array):
    x_begin_scaled, x_end_scaled, z_begin_scaled, z_end_scaled = get_array_indexes_2d(x_beg, x_end, z_beg, z_end, array.shape)
    return array[z_begin_scaled:z_end_scaled, x_begin_scaled:x_end_scaled]


def get_array_indexes_from_object_2d(obj, shape):
    return get_array_indexes_2d(obj.min_x, obj.max_x, obj.min_z, obj.max_z, shape)

def get_indexes_from_object(obj, array):
    return get_array_indexes(obj.min_x, obj.max_x, obj.min_z, obj.max_z, array.shape)


def is_slices_overlaps(first, second):
    is_x_overlaps = (first.min_x < second.min_x < first.max_x) or (first.min_x < second.max_x < first.max_x)
    is_y_overlaps = (first.min_z < second.min_z < first.max_z) or (first.min_z < second.max_z < first.max_z)
    return is_x_overlaps and is_y_overlaps


def slices_and(first, second):
    overlap_min_x = max(first.min_x, second.min_x)
    overlap_max_x = min(first.max_x, second.max_x)
    overlap_min_z = max(first.min_z, second.min_z)
    overlap_max_z = min(first.max_z, second.max_z)
    return overlap_min_x, overlap_max_x, overlap_min_z, overlap_max_z


def slices_or(first, second):
    or_min_x = min(first.min_x, second.min_x)
    or_max_x = max(first.max_x, second.max_x)
    or_min_z = min(first.min_z, second.min_z)
    or_max_z = max(first.max_z, second.max_z)
    return or_min_x, or_max_x, or_min_z, or_max_z



class PlotlyViewer(QtWebEngineWidgets.QWebEngineView):
    def __init__(self, fig, win_name, exec=True):
        from PyQt5.QtWidgets import QDesktopWidget

        # Create a QApplication instance or use the existing one if it exists
        self.app = QApplication.instance() if QApplication.instance() else QApplication(sys.argv)
        super().__init__()
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp.html"))
        self.resize(QDesktopWidget().availableGeometry().width(), QDesktopWidget().availableGeometry().height())

        # self.file_path = os.path.abspath(os.path.join(os.path.dirname('.'), "temp.html"))
        plotly.offline.plot(fig, filename=self.file_path, auto_open=False)
        self.load(QUrl.fromLocalFile(self.file_path))
        self.setWindowTitle(win_name)
        self.show()
        if exec:
            self.app.exec_()

    def closeEvent(self, event):
        os.remove(self.file_path)


def stone_slice_visualisation_in_window(stone_figure):
    plt.ion()

    # matplotlib.use('WebAgg')
    # matplotlib.use('Qt5Agg')

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    layout = [[sg.Text('Просмотр камня', auto_size_text=True)],
              [sg.Canvas(key='-CANVAS-')]]

    # create the form and show it without the plot
    window = sg.Window('', layout, finalize=True,
                       element_justification='center',
                       size=(QDesktopWidget().availableGeometry().width(),
                             QDesktopWidget().availableGeometry().height()),
                       location=(0, 0),
                       resizable=True)

    # add the plot to the window
    draw_figure(window['-CANVAS-'].TKCanvas, stone_figure)

    window.read()
    window.close()


def normalize(arr):
    if np.any(arr != 0):
        arr_min = np.min(arr)
    else:
        arr_min = 0
    try:
        return (arr - arr_min) / (np.max(arr) - arr_min)
    except Exception:
        return arr


def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded


def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z


def stone_vox3D_visualisation(cube, SliceThickness=1, x_thin=1, y_thin=1, st_angle=15, angle=75, cmap=cm.afmhot):

    if cube.shape[0] >= 4 and cube.shape[1] >= 4 and cube.shape[2] >= 4:
        dx = int(cube.shape[0] * 0.85)
        dy = int(cube.shape[1] * 0.85)
        dz = int(cube.shape[2] * 0.85)
    else:
        dx = cube.shape[0]
        dy = cube.shape[1]
        dz = cube.shape[2]

    cube1 = cube[:dx, :dy, :dz]
    cube = normalize(cube)
    cube1 = normalize(cube1)

    facecolors = cmap(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)
    facecolors1 = cmap(cube1)
    facecolors1[:, :, :, -1] = cube1
    facecolors1 = explode(facecolors1)

    filled = facecolors[:, :, :, -1] != 0
    filled1 = facecolors1[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    x1, y1, z1 = expand_coordinates(np.indices(np.array(filled1.shape) + 1))

    z = (z * x_thin if x_thin > 1 else z / x_thin)
    y = (y * y_thin if y_thin > 1 else y / y_thin)
    x = (x * SliceThickness if SliceThickness > 1 else x / SliceThickness)
    x1 = (x1 * x_thin if x_thin > 1 else x1 / x_thin)
    y1 = (y1 * y_thin if y_thin > 1 else y1 / y_thin)
    z1 = (z1 * SliceThickness if SliceThickness > 1 else z1 / SliceThickness)

    # fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    fig = plt.figure(figsize=(10, 10))
    # fig = plt.figure(figsize=(Length_stone/3,Height_stone/3))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    ax.view_init(st_angle, angle)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.voxels(x1, y1, z1, filled1, facecolors=facecolors1, shade=False)
    ax.view_init(st_angle, angle)

    if cmap == cm.afmhot:
        norm = matplotlib.colors.Normalize(vmin=160, vmax=1300)
        ticks_colors = [160, 300, 500, 800, 900, 1000, 1100, 1200, 1250, 1300]
        ticks_label = 'HU'
    else:
        norm = matplotlib.colors.Normalize(vmin=1.7, vmax=2.5)
        ticks_colors = [1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
        ticks_label = "гр/см3"
        # p = ax.voxels(x1, y1, z1, filled1, facecolors=facecolors1, shade=False)
    p = ax.voxels(x1, y1, z1, filled1, facecolors=facecolors1, shade=False)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(p)
    plt.colorbar(m, ax=ax, fraction=0.045, pad=0.1, ticks=ticks_colors, label=ticks_label)
    return fig


def stone_3proj_view(ds_array, frame_size_of_stone, realLength_stone, realHeight_stone,SliceThickness=1,
                     x_thin=1, y_thin=1, st_angle=15, angle=75, cmap=cm.afmhot):
    stone_HU_min = 0
    stone_HU_max = 1300
    vmin = stone_HU_min  # right_kidney_array.min()
    vmax = stone_HU_max  # right_kidney_array.max()
    steps = int((vmax - vmin) / 10)
    ticks = [i for i in range(vmin, vmax + steps, steps)]
    str_ticks = [i for i in ticks]

    # form coronal slice of stone
    sum_ligth = 0
    coronal_slice = 0
    max_ligth = 0
    if ds_array.shape[1] >= 1:
        for y in range(ds_array.shape[1]):
            if sum_ligth > max_ligth:
                max_ligth = sum_ligth
                coronal_slice = y
            sum_ligth = 0
            for z in range(ds_array.shape[0]):
                for x in range(ds_array.shape[2]):
                    sum_ligth += ds_array[z, y, x]
    else:
        coronal_slice = ds_array.shape[1] - 1

    # form sagittal slice of stone
    sum_ligth = 0
    sagittal_slice = 0
    max_ligth = 0
    if ds_array.shape[2] >= 1:
        for x in range(ds_array.shape[2]):
            if sum_ligth > max_ligth:
                max_ligth = sum_ligth
                sagittal_slice = x
            sum_ligth = 0
            for y in range(ds_array.shape[1]):
                for z in range(ds_array.shape[0]):
                    sum_ligth += ds_array[z, y, x]
    else:
        sagittal_slice = ds_array.shape[2] - 1

    # form axial slice of stone
    sum_ligth = 0
    axial_slice = 0
    max_ligth = 0
    if ds_array.shape[0] >= 1:
        for z in range(ds_array.shape[0]):
            if sum_ligth > max_ligth:
                max_ligth = sum_ligth
                axial_slice = z
            sum_ligth = 0
            for y in range(ds_array.shape[1]):
                for x in range(ds_array.shape[2]):
                    sum_ligth += ds_array[z, y, x]
    else:
        axial_slice = ds_array.shape[0] - 1

    # forn 3D model of stone
    stone_coronal = ds_array[:, coronal_slice, :]
    stone_sagittal = ds_array[:, :, sagittal_slice]
    stone_axial = ds_array[axial_slice, :, :]

    stone_coronal = cv2.resize(stone_coronal, frame_size_of_stone, interpolation=cv2.INTER_CUBIC)
    stone_sagittal = cv2.resize(stone_sagittal, frame_size_of_stone, interpolation=cv2.INTER_CUBIC)
    stone_axial = cv2.resize(stone_axial, frame_size_of_stone, interpolation=cv2.INTER_CUBIC)

    cube = ds_array[::-1, :, ::-1].T
    # cube = np.rot90(ds_array[:, :, :])
    cube = normalize(cube)
    facecolors = cmap(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)
    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    x = x * x_thin if x_thin > 1 else x / x_thin
    y = y * y_thin if y_thin > 1 else y / y_thin
    z = z * SliceThickness if SliceThickness > 1 else z / SliceThickness

    # plot figure
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(221)
    plt.title("Корональная проекция камня")
    plt.xlabel('Width, mm', fontsize=10)
    plt.ylabel('Height, mm', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    coronal_img = ax.imshow(stone_coronal, origin='upper', cmap=plt.cm.inferno,
                            extent=(0, realLength_stone, 0, realHeight_stone), vmin=vmin, vmax=vmax)
    # Create axis for colorbar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='10%', pad=0.05)
    # Create colorbar
    cbar = fig.colorbar(mappable=coronal_img, cax=cbar_ax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(str_ticks)

    ax = fig.add_subplot(222)
    plt.title("Саггитальная проекция камня")
    plt.xlabel('Width, mm', fontsize=10)
    plt.ylabel('Height, mm', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    sagittal_img = ax.imshow(stone_sagittal, origin='upper', cmap=plt.cm.inferno,
                             extent=(0, realLength_stone, 0, realHeight_stone), vmin=vmin, vmax=vmax)
    # Create axis for colorbar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='10%', pad=0.05)
    # Create colorbar
    cbar = fig.colorbar(mappable=sagittal_img, cax=cbar_ax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(str_ticks)

    ax = fig.add_subplot(223)
    plt.title("Аксиальная проекция камня")
    plt.xlabel('Width, mm', fontsize=10)
    plt.ylabel('Height, mm', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))

    axial_img = ax.imshow(stone_axial, origin='upper', cmap=plt.cm.inferno,
                          extent=(0, realLength_stone, 0, realHeight_stone), vmin=vmin, vmax=vmax)
    # Create axis for colorbar
    cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='10%', pad=0.05)
    # Create colorbar
    cbar = fig.colorbar(mappable=axial_img, cax=cbar_ax)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(str_ticks)

    ax = fig.add_subplot(224, projection='3d')
    plt.title("3D визуализация камня")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    ax.view_init(st_angle, angle)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    p = ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(p)
    # plt.colorbar(m, ax=ax, fraction=0.046, pad=0.1, ticks=ticks)
    plt.colorbar(m, ax=ax, fraction=0.02, pad=0.1, ticks=ticks, label='HU')

    # plt.show()
    return fig


def kidney_3D_visualisation(values, SliceThickness, PixelSpacingX, PixelSpacingY):
    X, Y, Z = np.mgrid[0:values.shape[0], 0:values.shape[1], 0:values.shape[2]]
    values[values <= 35] = 0
    values = values.flatten()
    """ colorsmaps
    ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
     'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
     'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
     'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
     'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
     'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
     'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
     'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
     'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
     'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
     'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
     'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
     'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
     'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
     'ylorrd']
    """
    data = go.Isosurface(
        x=(X * PixelSpacingX if PixelSpacingX > 1 else X / PixelSpacingX).flatten(),
        y=(Y * PixelSpacingY if PixelSpacingY > 1 else Y / PixelSpacingY).flatten(),
        z=(Z * SliceThickness if SliceThickness > 1 else Z / SliceThickness).flatten(),
        value=values.flatten(),
        isomin=25,
        isomax=159,
        colorscale='plasma',
        surface_count=5,
        colorbar_nticks=5,
        opacity=0.9,
    )

    fig = go.Figure(data)
    return fig


def stone_3D_visualisation(values, SliceThickness, PixelSpacingX, PixelSpacingY):
    X, Y, Z = np.mgrid[0:values.shape[0], 0:values.shape[1], 0:values.shape[2]]
    data = go.Isosurface(
        x=(X*PixelSpacingX if PixelSpacingX > 1 else X/PixelSpacingX).flatten(),
        y=(Y*PixelSpacingY if PixelSpacingY > 1 else Y/PixelSpacingY).flatten(),
        z=(Z*SliceThickness if SliceThickness > 1 else Z/SliceThickness).flatten(),
        value=values.flatten(),
        isomin=160,
        isomax=1350,
        colorscale='hot', # 'inferno',
        # surface=dict(show=True, count=1, fill=0.9),
        slices=go.isosurface.Slices(
            x=go.isosurface.slices.X(show=True, locations=[-0.5, 0.5]),
            y=go.isosurface.slices.Y(show=True, locations=[-0.5, 0.5]),
            z=go.isosurface.slices.Z(show=True, locations=[-0.5, 0.5])
        ),
        caps=go.isosurface.Caps(
            z=dict(show=False),
            x=dict(show=False),
            y=dict(show=False)
        ),
        # opacity=0.5, # needs to be small to see through all surfaces
        surface_count=8,
        colorbar_nticks=8,
        opacity=0.5,
        surface_fill=0.98
    )

    layout = go.Layout(
        margin=dict(t=0, l=0, b=0),
        scene=dict(
            camera=dict(
                eye=dict(
                    x=PixelSpacingX,
                    y=PixelSpacingY,
                    z=SliceThickness)
            )
        )
    )
    fig = go.Figure(data)
    return fig

def main(input_path):
    lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme \
        = cf.read_settings()
    tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)

    # noinspection PyMethodMayBeStatic
    class Parser:
        """ нужно ли парсить строку файла """
        def parse_condition(self, line):
            first_token = line.split(' ')[0]
            return first_token.isdigit()

        ''' преобразование строки в объект '''
        def line_transform(self, line):
            tokens = list(map(lambda x: float(x), line.split(' ')))
            return SliceDTO(
                label=tokens[0],
                x=tokens[1],
                z=tokens[2],
                w=tokens[3],
                h=tokens[4],
                conf=tokens[5]
            )

        ''' парсинг имени файла '''
        def filename_parser(self, filename):
            # return int(filename[5:8]) # для старой разметки имен файлов
            return int(filename[12:15])

        ''' парсинг файла '''
        def parse(self, path, filename):
            file = open(path + filename, 'r', encoding="ISO-8859-1")
            try:
                lines = [line for line in file.readlines() if self.parse_condition(line)]
                y = self.filename_parser(filename)
                layer = LayerDTO(y, list(map(self.line_transform, lines)))
                return layer
            except Exception:
                file.close()
                raise

    class kidney_type:
        normal = 0
        pieloectasy = 1

    def get_kidney_info(kidney):
        first = max_int_size
        last = min_int_size
        max_square = min_int_size
        max_square_index = 0
        max_square_center = (0, 0)
        max_w = min_int_size
        max_h = min_int_size
        size = 0
        cur_kidney_type = kidney_type.normal
        for (key, value) in filter(lambda x: x[1], kidney.items()):
            size += 1
            first = min(key, first)
            last = max(key, last)
            cur_square = value.w * value.h
            if cur_square > max_square:
                max_square = cur_square
                max_square_index = key
                max_square_center = (value.x, value.z)
                max_w = value.w
                max_h = value.h

            if value.label == left_kidney_pieloectasy or value.label == right_kidney_pieloectasy:
                cur_kidney_type = kidney_type.pieloectasy

        return ObjectParamsDto(
            x_center=max_square_center[0],
            z_center=max_square_center[1],
            w=max_w,
            h=max_h,
            number=max_square_index,
            first_index=first,
            last_index=last
        )

    ''' находится ли объект в ограничениях для правой почки '''
    def right_kidney_condition(slice):
        return is_right_kidney(slice) and is_in_right_kidney_constraints(slice)

    ''' находится ли объект в ограничениях для левой почки '''
    def left_kidney_condition(slice):
        return is_left_kidney(slice) and is_in_left_kidney_constraints(slice)

    ''' выбрать почку с максимальным правдоподобием '''
    def kidney_with_max_conf(lst):
        if not len(lst):
            return None
        return max(lst, key=lambda x: x.conf)

    ''' выбрать правую почку с максимальным правдоподобием '''
    def right_kidney_with_max_conf(lst):
        return kidney_with_max_conf(list(filter(right_kidney_condition, lst)))

    ''' выбрать левую почку с максимальным правдоподобием'''
    def left_kidney_with_max_conf(lst):
        return kidney_with_max_conf(list(filter(left_kidney_condition, lst)))

    def get_kidney_array(kidney_list, light_array):
        kidney_info = get_kidney_info(kidney_list)
        x_begin_scaled, x_end_scaled, z_begin_scaled, z_end_scaled = get_indexes_from_object(
            kidney_list[kidney_info.number], light_array)
        return light_array[z_begin_scaled:z_end_scaled, kidney_info.first_index:kidney_info.last_index,
               x_begin_scaled:x_end_scaled]

    def stone_clusterize(stone_dict):
        stones = []
        for (key, layer) in stone_dict.items():
            for cur_slice in layer:
                cur_stone = {key: [cur_slice]}
                overlaps_stones = []
                for (i, prev_stone) in enumerate(stones):
                    if key - 1 in prev_stone:
                        prev_layer = prev_stone[key - 1]
                        for prev_slice in prev_layer:
                            if is_in_other_slice(prev_slice, cur_slice) or is_in_other_slice(cur_slice,
                                                                                                         prev_slice):
                                overlaps_stones.append(prev_stone)
                                break
                for overlaps_stone in overlaps_stones:
                    for (prev_key, prev_layer) in overlaps_stone.items():
                        if prev_key in cur_stone:
                            cur_stone[prev_key] = [*cur_stone[prev_key], *prev_layer]
                        else:
                            cur_stone[prev_key] = prev_layer
                    stones.remove(overlaps_stone)

                stones.append(cur_stone)
        return stones

    def stone_info(stone_list):
        array = load_numpy_array()
        res = []
        for (i, stone) in enumerate(stone_list):
            first = max_int_size
            last = min_int_size
            max_light = min_int_size
            max_light_index = 0
            max_light_center = (0, 0)
            max_light_params = (0, 0)
            for (key, layer) in stone.items():
                first = min(key, first)
                last = max(key, last)
                for cur_slice in layer:
                    x_begin_scaled, x_end_scaled, z_begin_scaled, z_end_scaled = get_indexes_from_object(
                        cur_slice,
                        array)
                    cur_light = array[z_begin_scaled:z_end_scaled, key, x_begin_scaled:x_end_scaled].sum()
                    if cur_light > max_light:
                        max_light = cur_light
                        max_light_index = key
                        max_light_center = (cur_slice.x, cur_slice.z)
                        max_light_params = (cur_slice.w, cur_slice.h)
            res.append(ObjectParamsDto(
                x_center=max_light_center[0],
                z_center=max_light_center[1],
                w=max_light_params[0],
                h=max_light_params[1],
                number=max_light_index,
                first_index=first,
                last_index=last
            ))
            '''
            print(f"=== камень {i} ===")
            print(f"первый индекс {first}")
            print(f"последний индекс {last}")
            print(f"максимальная светимость {max_light} на снимке {max_light_index}")
            print(f"    с центром в {max_light_center[0]} {max_light_center[1]}")
            print(f"    с шириной {max_light_params[0]} и высотой {max_light_params[1]}")
            print()
            '''
        return res

    # define dir of output stones parameters
    def get_output_path():
        if not os.path.exists(stones_dir_path):
            os.makedirs(stones_dir_path)
            print('Создан каталог ', stones_dir_path)
        # else:
        #    shutil.rmtree(stones_dir_path)
        #    os.makedirs(stones_dir_path)
        return stones_dir_path

    # load array
    def load_numpy_array():
        ds_array = np.load(numpy_file_path).astype(np.int16)
        return ds_array

    def get_filename_param_csv():
        return filename_param_csv_path

    # get parameters of found stones
    def get_found_stones_param(index, kidney_position):
        if kidney_position == 'left':
            stone = left_stones_params[index]
        elif kidney_position == 'right':
            stone = right_stones_params[index]
        return stone.x_center, stone.z_center, stone.w, stone.h, stone.first_index, stone.last_index, stone.number

    # get parameters of numpy array
    def get_numpy_parameters(filename):
        param_num = []
        fieldnames = ["Study Date",
                      "Series Description",
                      "Patient's Name",
                      "Patient ID",
                      "Spacing Between Slices",
                      "Series Number",
                      "Start Slice Location",
                      "End Slice Location",
                      "Slice Thickness",
                      "Rows",
                      "Columns",
                      "Samples per Pixel",
                      "Pixel Spacing X",
                      "Pixel Spacing Y",
                      "Rescale Intercept",
                      "Rescale Slope",
                      "Shape Z, Y, X",
                      "Z",
                      "Y",
                      "X"
                      ]

        with open(filename, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for i in range(len(fieldnames)):
                    # print(row[fieldnames[i]])
                    param_num.append(row[fieldnames[i]])
        return param_num

    def stone_slice_visualisation(current_stone_array, med_slice, realLength_stone, realHeight_stone, kid_stone_index):
        ticks = [0, 160, 300, 500, 800, 900, 1000, 1100, 1200, 1250, 1300]
        ticks_labels = ['0', '160', '300', '500', '800', '900', '1000', '1100', '1200', '1250', '1300HU']
        # Create figure and add axis
        # fig = plt.figure(figsize=(frame_size_stone), dpi=100)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        plt.xlabel('Width, mm', fontsize=10)
        plt.ylabel('Height, mm', fontsize=10)
        plt.title(f'Stone {kid_stone_index} at {med_slice} slice')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
        # Show AFM image
        img = ax.imshow(current_stone_array, origin='upper', cmap=plt.cm.inferno,
                        extent=(0, realLength_stone, 0, realHeight_stone), vmin=0, vmax=1300)
        # Create axis for colorbar
        cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)
        # Create colorbar
        cbar = fig.colorbar(mappable=img, cax=cbar_ax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks_labels)
        return fig

    def calc_stone_parameters(index, kidney_position):
        global stone_param, index_kidney, norm_stone_image, frame_size_stone, ds_array, z_beg, z_end, start_slice, \
            end_slice, x_beg, x_end, med_slice, realLength_stone, realHeight_stone
        # filename_param_csv = get_filename_param_csv()
        param_numpy = get_numpy_parameters(get_filename_param_csv())
        ds_array = load_numpy_array()
        index_kidney = 'lk_' if kidney_position == 'left' else 'rk_'
        filename_stone_param = get_output_path() + 'stnpar_' + index_kidney + str(index) + '.txt'

        # StudyDate = param_numpy[0]
        # SeriesDescription = param_numpy[1]
        # PatientName = param_numpy[2]
        # PatientID = param_numpy[3]
        z_thin = float(param_numpy[4])
        # SeriesNumber = param_numpy[5]
        # StartSliceLocation = param_numpy[6]
        # EndSliceLocation = param_numpy[7]
        SliceThickness = float(param_numpy[8])
        # Rows = param_numpy[9]
        # Columns = param_numpy[10]
        # SamplesperPixel = param_numpy[11]
        x_thin = float(param_numpy[12])
        y_thin = float(param_numpy[13])
        # RescaleIntercept = param_numpy[14]
        # RescaleSlope = param_numpy[15]
        # ShapeZYX = param_numpy[16]
        z_ = int(param_numpy[17])
        # y_ = int(param_numpy[18])
        x_ = int(param_numpy[19])

        x_center, z_center, w, h, start_slice, end_slice, med_slice = get_found_stones_param(index, kidney_position)
        x_beg = int(x_ * (x_center - w / 2))
        x_end = int(x_ * (x_center + w / 2))
        z_beg = int(z_ * (z_center - h / 2))
        z_end = int(z_ * (z_center + h / 2))
        Length_stone = (x_end - x_beg)
        Height_stone = (z_end - z_beg)
        realLength_stone = Length_stone * x_thin
        realHeight_stone = Height_stone * z_thin
        x_frame = Length_stone
        y_frame = int(Length_stone * (realHeight_stone / realLength_stone))
        frame_size_stone = (x_frame, y_frame)
        stone_3d = ds_array[z_beg:z_end, start_slice:end_slice, x_beg:x_end]
        only_stone = []
        count_stone_vox = 0

        for z in range(stone_3d.shape[0]):
            for y in range(stone_3d.shape[1]):
                for x in range(stone_3d.shape[2]):
                    if stone_3d[z, y, x] > 160:
                        only_stone.append(stone_3d[z, y, x])
                        count_stone_vox += 1
                    else:
                        only_stone.append(0)
        only_stone = np.array(only_stone)
        only_stone = only_stone.reshape(stone_3d.shape)

        if np.any(only_stone != 0):
            max_HU = only_stone.max()
            min_HU = only_stone.max()
        else:
            max_HU = min_HU = 0

        # find min HU
        for z in range(only_stone.shape[0]):
            for y in range(only_stone.shape[1]):
                for x in range(only_stone.shape[2]):
                    if only_stone[z, y, x] != 0:
                        if only_stone[z, y, x] < min_HU:
                            min_HU = only_stone[z, y, x]

        # find average HU
        sum_HU = 0
        for z in range(only_stone.shape[0]):
            for y in range(only_stone.shape[1]):
                for x in range(only_stone.shape[2]):
                    if only_stone[z, y, x] != 0:
                        sum_HU += only_stone[z, y, x]
        ave_HU = (sum_HU / count_stone_vox) if count_stone_vox != 0 else 0

        # calc density of stones array
        dens_stone = []
        dens_stone_sum = 0
        for z in range(only_stone.shape[0]):
            for y in range(only_stone.shape[1]):
                for x in range(only_stone.shape[2]):
                    if only_stone[z, y, x] != 0:
                        dens_stone.append((only_stone[z, y, x] * 0.000485 + 1.539))
                        dens_stone_sum += (only_stone[z, y, x] * 0.000485 + 1.539)
                    else:
                        dens_stone.append(0)
        dens_stone = np.array(dens_stone)
        dens_stone = dens_stone.reshape(only_stone.shape)

        Length_stone = (x_end - x_beg) * x_thin / 10
        Height_stone = (z_end - z_beg) * z_thin / 10

        # calc volume of stone
        volume_unit_vox = x_thin * y_thin * z_thin
        volume_unit_sm = volume_unit_vox / 1000
        volume_stone = volume_unit_sm * count_stone_vox
        mass_stone = volume_unit_sm * dens_stone_sum
        volume_stone1 = volume_unit_sm * dens_stone.shape[0] * dens_stone.shape[1] * dens_stone.shape[2]
        if count_stone_vox > 0:
            ave_dens = dens_stone_sum / count_stone_vox
        else:
            ave_dens = 0

        # stone_param = []
        stone_param = [index,  # 0
                       med_slice,  # 1
                       Length_stone,  # 2
                       Height_stone,  # 3
                       volume_stone1 / (Length_stone * Height_stone),  # 4
                       volume_unit_vox,  # 5
                       volume_unit_sm,  # 6
                       volume_stone1,  # 7
                       volume_stone,  # 8
                       count_stone_vox,  # 9
                       mass_stone,  # 10
                       ave_dens,  # 11
                       ave_dens * volume_stone,  # 12
                       max_HU,  # 13
                       min_HU,  # 14
                       ave_HU,  # 15
                       x_beg,  # 16
                       x_end,  # 17
                       z_beg,  # 18
                       z_end,  # 19
                       start_slice,  # 20
                       end_slice,  # 21
                       realLength_stone,  # 22
                       realHeight_stone,  # 23
                       frame_size_stone]  # 24

        # save param`s of stone into file
        with open(filename_stone_param, 'w') as f:
            f.write(f'Параметры камня: \n')
            f.write(f'{list_text[19] if kidney_position == "right" else list_text[20]}, камень {index}, ')
            # f.write(f'срез {list_slices_of_stones_RK[index_of_stone] if kidney_key == "right" else list_slices_of_stones_LK[index_of_stone]}\n')
            f.write(
                f'размеры камня - {stone_param[2]:.2f} см Х {stone_param[3]:.2f} см Х {stone_param[4]:.2f} см \n')
            # s4 = f'объем 1 вокселя, мм3 - {stone_param[5]:.2f}, объем 1 вокселя, см3 - {stone_param[6]:.2f}\n'
            # s5 = f'объем пространства, см3 - {stone_param[7]:.2f}, объем камня, см3 - {stone_param[8]:.2f}\n'
            # s6 = f'количество вокселей в камне - {stone_param[9]}\n'
            f.write(f'масса камня, грамм -  {stone_param[10]:.2f}\n')
            f.write(f'средняя плотность, гр/см3 -  {stone_param[11]:.2f}\n')
            f.write(f'масса по средней плотности, грамм -  {stone_param[12]:.2f}\n')
            f.write(f'максимальная плотность по HU = {stone_param[13]}\n')
            f.write(f'минимальная плотность по HU = {stone_param[14]}\n')
            f.write(f'средняя плотность по HU = {stone_param[15]:.0f}\n')

        # calc new reduced density of stones array
        new_dens_stone = []
        for z in range(dens_stone.shape[0]):
            for y in range(dens_stone.shape[1]):
                for x in range(dens_stone.shape[2]):
                    if dens_stone[z, y, x] == 0:
                        new_dens_stone.append(0)
                    elif (dens_stone[z, y, x] < 1.75) and (dens_stone[z, y, x] > 0):
                        new_dens_stone.append(1.7)
                    elif (dens_stone[z, y, x] >= 1.75) and (dens_stone[z, y, x] < 1.85):
                        new_dens_stone.append(1.8)
                    elif (dens_stone[z, y, x] >= 1.85) and (dens_stone[z, y, x] < 1.95):
                        new_dens_stone.append(1.9)
                    elif (dens_stone[z, y, x] >= 1.95) and (dens_stone[z, y, x] < 2.05):
                        new_dens_stone.append(2.0)
                    elif (dens_stone[z, y, x] >= 2.05) and (dens_stone[z, y, x] < 2.15):
                        new_dens_stone.append(2.1)
                    elif (dens_stone[z, y, x] >= 2.15) and (dens_stone[z, y, x] < 2.25):
                        new_dens_stone.append(2.2)
                    elif (dens_stone[z, y, x] >= 2.25) and (dens_stone[z, y, x] < 2.35):
                        new_dens_stone.append(2.3)
                    elif dens_stone[z, y, x] >= 2.35:
                        new_dens_stone.append(2.4)
        new_dens_stone = np.array(new_dens_stone)
        new_dens_stone = new_dens_stone.reshape(dens_stone.shape)

        np.save(get_output_path() + 'st_' + index_kidney + str(index), only_stone)

        # visualisation

        # plot slice stone image
        stone_image = ds_array[z_beg:z_end, med_slice, x_beg:x_end]
        norm_stone_image = cv2.resize(stone_image, frame_size_stone)
        stone_slice_visualisation(norm_stone_image, med_slice, realLength_stone, realHeight_stone,
                                  index_kidney + str(index))
        image_name = get_output_path() + 'stone_' + index_kidney + str(index) + '.png'
        plt.savefig(image_name, bbox_inches='tight', transparent=True, format='png')

        # plot 3D image of density of stones
        if only_stone.shape[0] >= 3 and only_stone.shape[1] >= 3 and only_stone.shape[2] >= 3:
            stone_vox3D_visualisation(only_stone[::-1, ::1, ::-1].T, SliceThickness, x_thin, y_thin,  12, 75)
            stone_image_name = get_output_path() + '/stone' + index_kidney + str(index) + '_1' + '.png'
            plt.savefig(stone_image_name, transparent=True, bbox_inches='tight', format='png')

        # plot 3D image of reduced density of stones
        if new_dens_stone.shape[0] >= 3 and new_dens_stone.shape[1] >= 3 and new_dens_stone.shape[2] >= 3:
            stone_vox3D_visualisation(new_dens_stone[::-1, ::1, ::-1].T, SliceThickness, x_thin, y_thin,
                                          12, 75, cm.Set1)
            stone_image_name = get_output_path() + '/stone' + index_kidney + str(index) + '_2' + '.png'
            plt.savefig(stone_image_name, transparent=True, bbox_inches='tight', format='png')

        # here you need to insert the output in PDF of the parameters of only current stone with pictures !!!!!!

        return stone_param

    def form_string(file_name):  # read text file with stone parameters
        text_param_stone = ''
        if Path(file_name).is_file():
            try:
                with open(file_name, "rt") as f:
                    text_param_stone = f.read()
            except Exception:
                text_param_stone = list_text[3]
        return text_param_stone

    input_path = str(input_path) + '/'
    labels_dir_path = input_path + 'detect/labels/'
    stones_dir_path = input_path + 'stones/'

    # ID = str('/' + Path(input_path).parts[3][0:4])
    # ID = gi.patient_ID
    # filename_param_csv_path = input_path + ID + 'arrayinfo.csv'
    filename_param_csv_path = input_path + [f for f in os.listdir(input_path) if f.lower().endswith('.csv')][0]
    param_numpy = get_numpy_parameters(get_filename_param_csv())

    labels_names = listdir(labels_dir_path)
    ID = param_numpy[3]
    SliceThickness = param_numpy[8]
    x_thin = param_numpy[12]
    y_thin = param_numpy[13]
    numpy_file_path = input_path + str(ID) + 'array.npy'
    light_array = np.load(numpy_file_path).astype(np.int16)
    parser = Parser()

    labels_list = sorted(map(lambda x: parser.parse(labels_dir_path, x), labels_names), key=lambda x: x.y)

    right_kidney_list = dict([(label.y, right_kidney_with_max_conf(label.slice_list)) for label in labels_list])
    left_kidney_list = dict([(label.y, left_kidney_with_max_conf(label.slice_list)) for label in labels_list])

    left_kidney_array = get_kidney_array(left_kidney_list, light_array)
    right_kidney_array = get_kidney_array(right_kidney_list, light_array)
    np.save(get_output_path() + 'LK', left_kidney_array)  # save left kidney numpy array
    np.save(get_output_path() + 'RK', right_kidney_array)  # save right kidney numpy array

    all_stones = dict([(label.y, list(filter(is_stone, label.slice_list))) for label in labels_list])
    stones_with_right_kidney = dict(filter(lambda x: right_kidney_list[x[0]], all_stones.items()))
    stones_with_left_kidney = dict(filter(lambda x: left_kidney_list[x[0]], all_stones.items()))

    for i in stones_with_right_kidney:
        stones_with_right_kidney[i] = list(
            filter(lambda x: is_in_other_slice(x, right_kidney_list[i]), stones_with_right_kidney[i]))
    for i in stones_with_left_kidney:
        stones_with_left_kidney[i] = list(
            filter(lambda x: is_in_other_slice(x, left_kidney_list[i]), stones_with_left_kidney[i]))

    left_stones_params = stone_info(stone_clusterize(stones_with_left_kidney))
    right_stones_params = stone_info(stone_clusterize(stones_with_right_kidney))

    list_slices_of_stones_RK = []
    list_slices_of_stones_LK = []

    n = len(right_stones_params) + len(left_stones_params)
    i_b = 0
    kidney_key = 'right'
    layout_bar = [
        [sg.Text(window_heads[5]+'....')],
        [sg.Text(f'{list_text[19] if kidney_key == "right" else list_text[20]}', key='-kidney')],
        [sg.ProgressBar(n, orientation='h', size=(50, 10), border_width=1, key='-PROG-')],
    ]

    # create the Window
    window_bar = sg.Window(window_heads[5], layout_bar, disable_close=True, no_titlebar=False, finalize=True)
    RS_params = []
    LS_params = []

    # calculation stones parameters
    # for right kidney
    for i in range(len(right_stones_params)):
        kidney_key = 'right'
        window_bar['-kidney'].update(list_text[19] if kidney_key == "right" else list_text[20])
        window_bar['-PROG-'].update(i_b + 1)
        i_b += 1
        calc_stone_parameters(i, kidney_key)
        list_slices_of_stones_RK.append(str(stone_param[1]))
        RS_params.append(stone_param)

    # for left kidney
    for i in range(len(left_stones_params)):
        kidney_key = 'left'
        window_bar['-kidney'].update(list_text[19] if kidney_key == "right" else list_text[20])
        window_bar['-PROG-'].update(i_b + 1)
        i_b += 1
        calc_stone_parameters(i, kidney_key)
        list_slices_of_stones_LK.append(str(stone_param[1]))
        LS_params.append(stone_param)

    window_bar.close()
    plt.close('all')


    # here you need to insert the output in PDF of the parameters of the stones !!!!!!
    pdfFileName = pw.create_PDF(stones_dir_path, RS_params, LS_params, param_numpy)


    filenames_only = [f for f in os.listdir(input_path + 'detect/') if f.lower().endswith('.png')]
    index_of_stone = 0
    if len(right_stones_params) > 0:
        kidney_key = 'right'
    elif len(left_stones_params) > 0:
        kidney_key = 'left'

    # list stones for window
    r_list_stones = [list_text[18] + ' № ' + str(i) for i in range(len(right_stones_params))]
    l_list_stones = [list_text[18] + ' № ' + str(i) for i in range(len(left_stones_params))]
    list_stones = r_list_stones if kidney_key == 'right' else l_list_stones

    index_kidney = 'lk_' if kidney_key == 'left' else 'rk_'
    param_file_name = stones_dir_path + 'stnpar_' + index_kidney + str(index_of_stone) + '.txt'  # filename text param

    stone_viewer_column = [
        [sg.Text(text=list_text[21], size=(60, 1), key="INDEX OF SLICE")],
        [sg.Image(subsample=1, key="-IMAGE-")],
        [sg.Image(key="-IMAGE0-", enable_events=True)],
        [sg.Multiline(default_text=form_string(param_file_name), size=(35, 11),
                      enable_events=True, key="-PARAM-")],
        [sg.Button(button_text="3D модель почки", size=(10, 2), key="-KIDNEY VIEW-", enable_events=True),
         sg.Button(button_text="3D модель камня", size=(10, 2), key="-STONE VIEW-", enable_events=True)]
    ]

    stone_vox3D_view = [
        [sg.Image(key="-IMAGE1-")],
        [sg.Image(key="-IMAGE2-")],
    ]

    tab_right_kidney = [[]]
    tab_left_kidney = [[]]
    tab_group_layout = [[sg.Tab(list_text[19], tab_right_kidney,
                                key="-select kidney right-",
                                disabled=True if len(right_stones_params) == 0 else False),
                         sg.Tab(list_text[20], tab_left_kidney,
                                key="-select kidney left-",
                                disabled=True if len(left_stones_params) == 0 else False),
                         ]]

    stone_list_column = [
        [sg.TabGroup(tab_group_layout, enable_events=True, key='-TAB GROUP-')],
        [sg.Listbox(values=list_stones, horizontal_scroll=True, enable_events=True, size=(30, 25),
                    key="-STONE LIST-")],
        [sg.Button(button_text=buttons[6], size=(8, 1)),
         sg.Button(button_text=buttons[7], size=(8, 1)),
         sg.Button(button_text=buttons[8], size=(8, 1)),
         sg.Button(button_text=buttons[9], size=(8, 1))],
        [sg.Button(button_text='Печать параметров камней', size=(30, 1), key='-PRINT_INFO_PDF-')],
        [sg.Cancel(button_text=buttons[5], size=(30, 1), button_color='teal')],
    ]

    layout_stones = [
        [sg.Column(stone_list_column, vertical_alignment='top'),
         sg.Column(stone_viewer_column, vertical_alignment='top'),
         sg.Column(stone_vox3D_view, vertical_alignment='top'),
         ],
    ]

    window_view_stone = sg.Window(window_heads[4], layout_stones, use_default_focus=True,
                                  return_keyboard_events=True, finalize=True,
                                  location=(0, 0), resizable=True,
                                  size=(QDesktopWidget().availableGeometry().width(),
                                        QDesktopWidget().availableGeometry().height()),
                                  no_titlebar=False)

    window_view_stone["-STONE LIST-"].update(values=list_stones, set_to_index=0, scroll_to_index=0)
    window_view_stone["-PARAM-"].update(form_string(param_file_name))
    window_view_stone["INDEX OF SLICE"].update(list_text[21] + ' № ' + list_slices_of_stones_RK[
        index_of_stone] if kidney_key == 'right' else list_text[21] + ' № ' + list_slices_of_stones_LK[index_of_stone])

    slice_image_name = filenames_only[index_of_stone][:-7] + list_slices_of_stones_RK[index_of_stone] + '.png' \
        if kidney_key == 'right' else filenames_only[index_of_stone][:-7] + list_slices_of_stones_LK[
        index_of_stone] + '.png'
    filename_of_stone_0 = get_output_path() + 'stone_' + index_kidney + str(index_of_stone) + '.png'
    filename_of_stone_1 = get_output_path() + 'stone' + index_kidney + str(index_of_stone) + '_1.png'
    filename_of_stone_2 = get_output_path() + 'stone' + index_kidney + str(index_of_stone) + '_2.png'

    window_view_stone["-IMAGE-"].update(input_path + 'detect/' + slice_image_name, subsample=2)
    window_view_stone["-IMAGE0-"].update(filename_of_stone_0 if os.path.isfile(filename_of_stone_0) else None,
                                         subsample=2)
    window_view_stone["-IMAGE1-"].update(filename_of_stone_1 if os.path.isfile(filename_of_stone_1) else None,
                                         subsample=1)
    window_view_stone["-IMAGE2-"].update(filename_of_stone_2 if os.path.isfile(filename_of_stone_2) else None,
                                         subsample=1)
    window_view_stone.maximize()

    while True:
        event, values = window_view_stone.read()
        if event == '-TAB GROUP-' and values['-TAB GROUP-'] == "-select kidney right-":
            kidney_key = 'right'
            list_stones = r_list_stones
            index_kidney = 'rk_'
            index_of_stone = 0

        elif event == '-TAB GROUP-' and values['-TAB GROUP-'] == "-select kidney left-":
            kidney_key = 'left'
            list_stones = l_list_stones
            index_kidney = 'lk_'
            index_of_stone = 0

        elif event == "-STONE LIST-":
            index_of_stone = list_stones.index(values["-STONE LIST-"][0])

        elif event in (buttons[8], 'Down:40', 'Next:34') and index_of_stone < len(list_stones) - 1:
            index_of_stone += 1

        elif event in (buttons[7], 'Up:38', 'Prior:33') and index_of_stone > 0:
            index_of_stone -= 1

        elif event == buttons[6]:
            index_of_stone = 0

        elif event == buttons[9]:
            index_of_stone = len(list_stones) - 1

        elif event == "-KIDNEY VIEW-":
            ind_kid = 'RK' if kidney_key == 'right' else 'LK'
            SliThick = param_numpy[8]
            xthin = param_numpy[12]
            ythin = param_numpy[13]
            values = np.load(get_output_path() + ind_kid + '.npy').astype(np.int32)
            PlotlyViewer(kidney_3D_visualisation(values[::-2, ::2, ::-2].T, float(SliThick), float(xthin),
                                                         float(ythin)),
                             list_text[19] if kidney_key == 'right' else list_text[20])

        elif event == "-STONE VIEW-":
            index_kidney = 'rk_' if kidney_key == 'right' else 'lk_'
            curr_stone = np.load(get_output_path() + 'st_' + index_kidney + str(index_of_stone) + '.npy').astype(
                np.int32)
            SliThick = param_numpy[8]
            xthin = param_numpy[12]
            ythin = param_numpy[13]
            PlotlyViewer(stone_3D_visualisation(curr_stone[::-1, ::1, ::-1].T, float(SliThick), float(xthin),
                                                        float(ythin)),
                             f'{list_text[18]} № {index_of_stone}')

        elif event == "-PRINT_INFO_PDF-":
            pw.read_n_print_pdf(pdfFileName)

        elif event == "-IMAGE0-":
            current_stone_param = RS_params[index_of_stone] if kidney_key == 'right' else \
                LS_params[index_of_stone]
            '''
            stone_param = [index,  # 0
                           med_slice,  # 1
                           Length_stone,  # 2
                           Height_stone,  # 3
                           volume_stone1 / (Length_stone * Height_stone),  # 4
                           volume_unit_vox,  # 5
                           volume_unit_sm,  # 6
                           volume_stone1,  # 7
                           volume_stone,  # 8
                           count_stone_vox,  # 9
                           mass_stone,  # 10
                           ave_dens,  # 11
                           ave_dens * volume_stone,  # 12
                           max_HU,  # 13
                           min_HU,  # 14
                           ave_HU,  # 15
                           x_beg,  # 16
                           x_end,  # 17
                           z_beg,  # 18
                           z_end,  # 19
                           start_slice,  # 20
                           end_slice]  # 21
            '''
            realLength = current_stone_param[22]
            realHeight = current_stone_param[23]
            frame_size_stone = current_stone_param[24]

            curr_stone = np.load(get_output_path() + 'st_' + index_kidney + str(index_of_stone) + '.npy').astype(
                np.int16)
            curr_stone[curr_stone < 160] = 0
            fig_stone = stone_3proj_view(curr_stone, frame_size_stone, realLength, realHeight,
                                 float(x_thin), float(y_thin), float(SliceThickness))
            stone_slice_visualisation_in_window(fig_stone)
            plt.close('all')

        elif event in (sg.WIN_CLOSED, buttons[5], 'Escape:27'):
            window_view_stone.close()
            break
        # print(stone_param[1])

        param_file_name = stones_dir_path + 'stnpar_' + index_kidney + str(index_of_stone) + '.txt'

        window_view_stone["-STONE LIST-"].update(values=list_stones, set_to_index=index_of_stone,
                                                 scroll_to_index=index_of_stone)
        window_view_stone["-PARAM-"].update(form_string(param_file_name))
        window_view_stone["INDEX OF SLICE"].update(list_text[21] + ' № ' + list_slices_of_stones_RK[
            index_of_stone] if kidney_key == 'right' else list_text[21] + ' № ' + list_slices_of_stones_LK[
            index_of_stone])

        slice_image_name = filenames_only[index_of_stone][:-7] + list_slices_of_stones_RK[index_of_stone] + '.png' \
            if kidney_key == 'right' else filenames_only[index_of_stone][:-7] + list_slices_of_stones_LK[
            index_of_stone] + '.png'
        filename_of_stone_0 = get_output_path() + 'stone_' + index_kidney + str(index_of_stone) + '.png'
        filename_of_stone_1 = get_output_path() + 'stone' + index_kidney + str(index_of_stone) + '_1.png'
        filename_of_stone_2 = get_output_path() + 'stone' + index_kidney + str(index_of_stone) + '_2.png'

        window_view_stone["-IMAGE-"].update(input_path + 'detect/' + slice_image_name, subsample=2)
        window_view_stone["-IMAGE0-"].update(filename_of_stone_0 if os.path.isfile(filename_of_stone_0) else None,
                                             subsample=2)
        window_view_stone["-IMAGE1-"].update(filename_of_stone_1 if os.path.isfile(filename_of_stone_1) else None,
                                             subsample=1)
        window_view_stone["-IMAGE2-"].update(filename_of_stone_2 if os.path.isfile(filename_of_stone_2) else None,
                                             subsample=1)


if __name__ == "__main__":
    main(input_path=gi.get_images_path('/SUD/'))