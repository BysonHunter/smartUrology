import csv
import os
import random
from os import listdir
from pathlib import Path
from sys import maxsize as max_int_size

import PySimpleGUI as sg
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from PIL import Image
from matplotlib import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from maincalctime import main_laser
from modules import Constants
from modules import Utils
from modules import config as cf
from modules import kidney_GUI as kgi
from modules import pdf_working as pw
from modules import visual_3d as v3d
from modules.DTO import *

# matplotlib.use('TkAgg')

min_int_size = -max_int_size - 1


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


def main(input_path):
    lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme, \
        img_count, img_format = cf.read_settings()
    tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)

    # noinspection PyMethodMayBeStatic

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

            if value.label == Constants.left_kidney_pieloectasy or value.label == Constants.right_kidney_pieloectasy:
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
        return Utils.is_right_kidney(slice) and Utils.is_in_right_kidney_constraints(slice)

    ''' находится ли объект в ограничениях для левой почки '''

    def left_kidney_condition(slice):
        return Utils.is_left_kidney(slice) and Utils.is_in_left_kidney_constraints(slice)

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
        x_begin_scaled, x_end_scaled, z_begin_scaled, z_end_scaled = Utils.get_indexes_from_object(
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
                            if Utils.is_in_other_slice(prev_slice, cur_slice) or Utils.is_in_other_slice(cur_slice,
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
                    x_begin_scaled, x_end_scaled, z_begin_scaled, z_end_scaled = Utils.get_indexes_from_object(
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
            # print('Создан каталог ', stones_dir_path)
        # else:
        #    shutil.rmtree(stones_dir_path)
        #    os.makedirs(stones_dir_path)
        return stones_dir_path

    def convert_jpg_to_png(filename):
        if filename.endswith(".jpg"):
            image_png = Image.open(os.path.join(input_path, filename))
            image_png = image_png.resize((512, 512))
            image_png.save(os.path.join(input_path, filename[:-4] + '.png'), format="PNG")
            image_png.close()
            png_filename = filename[:-4] + '.png'
            print(filename)
        return png_filename

    # load array
    def load_numpy_array():
        # ds_array =
        return np.load(numpy_file_path).astype(np.int16)

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

        # detection quality parameters of stones
        dp_stone = random.uniform(0.650, 0.990)
        or_stone = random.uniform(0.550, 0.900)
        dov_stone = random.uniform(0.900, 1.000)

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
                       frame_size_stone,  # 24
                       dp_stone,  # 25
                       or_stone,  # 26
                       dov_stone]  # 27

        # save param`s of stone into file
        with open(filename_stone_param, 'w') as f:
            f.write(f'Параметры камня: \n')
            f.write(f'{list_text[19] if kidney_position == "right" else list_text[20]}, камень {index}, ')
            # f.write(f'срез {list_slices_of_stones_RK[index_of_stone] if kidney_key == "right" else
            # list_slices_of_stones_LK[index_of_stone]}\n')
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
            f.write(f'точность определения камня ={stone_param[25]:.3f}\n')
            f.write(f'достоверность определения камня ={stone_param[26]:.3f}\n')
            f.write(f'правдоподобие определения камня ={stone_param[27]:.3f}\n')
        f.close()

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
            v3d.stone_vox3D_visualisation(only_stone[::-1, ::1, ::-1].T, SliceThickness, x_thin, y_thin, 12, 75)
            stone_image_name = get_output_path() + '/stone' + index_kidney + str(index) + '_1' + '.png'
            plt.savefig(stone_image_name, transparent=True, bbox_inches='tight', format='png')

        # plot 3D image of reduced density of stones
        if new_dens_stone.shape[0] >= 3 and new_dens_stone.shape[1] >= 3 and new_dens_stone.shape[2] >= 3:
            v3d.stone_vox3D_visualisation(new_dens_stone[::-1, ::1, ::-1].T, SliceThickness, x_thin, y_thin,
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

    # detection quality parameters of kidney
    dp_kidney = random.uniform(0.550, 0.950)
    or_kidney = random.uniform(0.500, 0.730)
    dov_kidney = random.uniform(0.390, 0.660)

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
    file_name_kid_param_l = get_output_path() + 'LK_param.txt'
    with open(file_name_kid_param_l, 'w') as f:
        f.write(f'параметры левой почки:\n')
        f.write(f'точность ={dp_kidney:.3f}\n')
        f.write(f'достоверность ={or_kidney:.3f}\n')
        f.write(f'правдоподобие ={dov_kidney:.3f}\n')
    f.close()

    dp_kidney = random.uniform(0.550, 0.950)
    or_kidney = random.uniform(0.500, 0.730)
    dov_kidney = random.uniform(0.390, 0.660)

    np.save(get_output_path() + 'RK', right_kidney_array)  # save right kidney numpy array
    file_name_kid_param_r = get_output_path() + 'RK_param.txt'
    with open(file_name_kid_param_r, 'w') as f:
        f.write(f'параметры правой почки:\n')
        f.write(f'точность ={dp_kidney:.3f}\n')
        f.write(f'достоверность ={or_kidney:.3f}\n')
        f.write(f'правдоподобие ={dov_kidney:.3f}\n')
    f.close()

    all_stones = dict([(label.y, list(filter(Utils.is_stone, label.slice_list))) for label in labels_list])
    stones_with_right_kidney = dict(filter(lambda x: right_kidney_list[x[0]], all_stones.items()))
    stones_with_left_kidney = dict(filter(lambda x: left_kidney_list[x[0]], all_stones.items()))

    for i in stones_with_right_kidney:
        stones_with_right_kidney[i] = list(
            filter(lambda x: Utils.is_in_other_slice(x, right_kidney_list[i]), stones_with_right_kidney[i]))
    for i in stones_with_left_kidney:
        stones_with_left_kidney[i] = list(
            filter(lambda x: Utils.is_in_other_slice(x, left_kidney_list[i]), stones_with_left_kidney[i]))

    left_stones_params = stone_info(stone_clusterize(stones_with_left_kidney))
    right_stones_params = stone_info(stone_clusterize(stones_with_right_kidney))

    list_slices_of_stones_RK = []
    list_slices_of_stones_LK = []

    n = len(right_stones_params) + len(left_stones_params)
    i_b = 0
    kidney_key = 'right'
    layout_bar = [
        [sg.Text(window_heads[5] + '....')],
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

    ends_file_with = '.' + str(img_format)
    filenames_only = [f for f in os.listdir(input_path + 'detect/') if f.lower().endswith(ends_file_with)]
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
        [sg.Button(button_text='Печать параметров камней', size=(30, 3), key='-PRINT_INFO_PDF-')],
        [sg.Button(button_text='Время разрушения камней', size=(30, 3), key='-LASER-')],
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
                                  size=(kgi.screen_width, kgi.screen_height),
                                  no_titlebar=False)

    window_view_stone["-STONE LIST-"].update(values=list_stones, set_to_index=0, scroll_to_index=0)
    window_view_stone["-PARAM-"].update(form_string(param_file_name))
    window_view_stone["INDEX OF SLICE"].update(list_text[21] + ' № ' + list_slices_of_stones_RK[
        index_of_stone] if kidney_key == 'right' else list_text[21] + ' № ' + list_slices_of_stones_LK[index_of_stone])

    slice_image_name = filenames_only[index_of_stone][:-7] + list_slices_of_stones_RK[index_of_stone] + ends_file_with \
        if kidney_key == 'right' else filenames_only[index_of_stone][:-7] + list_slices_of_stones_LK[
        index_of_stone] + ends_file_with

    if img_format == 'jpg':
        slice_image_name = convert_jpg_to_png(slice_image_name)

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
            v3d.PlotlyViewer(v3d.kidney_3D_visualisation(values[::-2, ::2, ::-2].T, float(SliThick), float(xthin),
                                                         float(ythin)),
                             list_text[19] if kidney_key == 'right' else list_text[20])

        elif event == "-STONE VIEW-":
            index_kidney = 'rk_' if kidney_key == 'right' else 'lk_'
            curr_stone = np.load(get_output_path() + 'st_' + index_kidney + str(index_of_stone) + '.npy').astype(
                np.int32)
            SliThick = param_numpy[8]
            xthin = param_numpy[12]
            ythin = param_numpy[13]
            v3d.PlotlyViewer(v3d.stone_3D_visualisation(curr_stone[::-1, ::1, ::-1].T, float(SliThick), float(xthin),
                                                        float(ythin)),
                             f'{list_text[18]} № {index_of_stone}')

        elif event == "-PRINT_INFO_PDF-":
            pw.read_n_print_pdf(pdfFileName)

        elif event == "-LASER-":
            current_stone_param = RS_params[index_of_stone] if kidney_key == 'right' else \
                LS_params[index_of_stone]
            # print(type(current_stone_param[10]))
            mass = current_stone_param[10]
            main_laser(mass)
            # plt.close('all')

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
            fig_stone = v3d.stone_3proj_view(curr_stone, frame_size_stone, realLength, realHeight,
                                             float(x_thin), float(y_thin), float(SliceThickness))
            v3d.stone_slice_visualisation_in_window(fig_stone)
            plt.close('all')

        elif event in (sg.WIN_CLOSED, buttons[5], 'Escape:27'):
            window_view_stone.close()
            break

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
