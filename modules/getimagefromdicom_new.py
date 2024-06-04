import csv
import datetime
import os
# from matplotlib import pyplot
from pydicom import dcmread
from pathlib import Path
import PySimpleGUI as sg
import cv2
import numpy as np
import pydicom
from modules import config as cf
from modules import kidney_GUI as km

# start_pos = 220
# stop_pos = 370
now = datetime.datetime.now()

lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme,  \
    img_count, img_format = cf.read_settings()
tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)

if int(img_count) < 100:
    img_count = 100

patient_ID = str()


def get_dicom_path(init_folder):
    # import fnmatch
    dicom_path = ''

    folder = sg.popup_get_folder(list_text[8],  # no_window=True,
                                 default_path=init_folder,
                                 initial_folder=init_folder,
                                 title=titles[0],
                                 # history=True, history_setting_filename='hist_infold.txt',
                                 modal=True
                                 )
    if folder is not None:
        folder = folder + '/'
        dicom_files = [f for f in os.listdir(folder) if
                   (f.endswith('dcm')) or (os.path.isfile(folder + f) and f == 'DICOMDIR')]

        if len(dicom_files) >= 1:
            dicom_path = folder
    #print(folder, '\n', dicom_files)
    return dicom_path


def get_images_path(start_folder):
    init_folder = start_folder
    images_path = ''
    folder = sg.popup_get_folder(list_text[0],
                                 default_path=init_folder,
                                 initial_folder=init_folder,
                                 # history=True, history_setting_filename='hist_outfold.txt',
                                 title=titles[1])
    if folder:
        images_path = folder
    return images_path


def selectSeries(series_numbers, SeriesDescription, CountOfImages):
    chanse = []
    for i in range(len(series_numbers)):
        chanse.append(f'Serie: {series_numbers[i]}, Descr: {SeriesDescription[i]}, SOP: {CountOfImages[i]}')

    series_select_column = [
        [sg.Listbox(
            values=chanse,
            enable_events=True,
            size=(50, len(chanse)),
            key='-SERIES LIST-',
        )],
        [sg.OK('Select Serie', key='OK')]
    ]
    selectedSerieNumber = series_numbers[0]
    index_of_serie = 0
    window_select_series = sg.Window('Select', layout=series_select_column,
                                     # size=screensize,
                                     return_keyboard_events=True, finalize=True,
                                     # location=(0, 0),
                                     resizable=True,
                                     no_titlebar=False)

    window_select_series['-SERIES LIST-'].update(chanse, set_to_index=index_of_serie, scroll_to_index=0)

    while True:
        event, values = window_select_series.read()

        if event in ('MouseWheel:Down', 'Down:40', 'Next:34') and index_of_serie < len(chanse) - 1:
            index_of_serie += 1
            window_select_series['-SERIES LIST-'].update(set_to_index=index_of_serie, scroll_to_index=index_of_serie)

        elif event in ('MouseWheel:Up', 'Up:38', 'Prior:33') and index_of_serie > 0:
            index_of_serie -= 1
            window_select_series['-SERIES LIST-'].update(set_to_index=index_of_serie, scroll_to_index=index_of_serie)

        elif event in (sg.WIN_CLOSED, 'OK', 'Escape:27'):
            selectedSerieNumber = series_numbers[index_of_serie]
            window_select_series.close()
            break

        window_select_series['-SERIES LIST-'].update(chanse, set_to_index=index_of_serie,
                                                     scroll_to_index=index_of_serie)
    return selectedSerieNumber


def readDICOMDIR(dicomdirpath):
    global image_filenames
    SeriesNumbers = []
    SeriesDescription = []
    CountOfImages = []
    slices = []
    new_slices = []
    root_dir = Path(dicomdirpath).resolve().parent
    ds = dcmread(dicomdirpath)

    # Iterate through the PATIENT records
    for patient in ds.patient_records:

        # Find all the STUDY records for the patient
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            # Find all the SERIES records in the study
            all_series = [
                ii for ii in study.children if ii.DirectoryRecordType == "SERIES"
            ]
            for series in all_series:
                # Find all the IMAGE records in the series
                images = [
                    ii for ii in series.children
                    if ii.DirectoryRecordType == "IMAGE"
                ]

                descr = getattr(
                    series, "SeriesDescription", None
                )
                if len(descr) >= 0:
                    SeriesNumbers.append(series.SeriesNumber)
                    SeriesDescription.append(descr)
                    CountOfImages.append(len(images))

    selectedSerieNumber = selectSeries(SeriesNumbers, SeriesDescription, CountOfImages)

    for patient in ds.patient_records:
        studies = [
            ii for ii in patient.children if ii.DirectoryRecordType == "STUDY"
        ]
        for study in studies:
            all_series = [
                ii for ii in study.children if
                ii.DirectoryRecordType == "SERIES" and ii.SeriesNumber == selectedSerieNumber
            ]
            for series in all_series:
                image_records = series.children
                image_filenames = [os.path.join(root_dir, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

    i = 0
    layout_bar = [
        [sg.Text(f'{list_text[11]} {root_dir}')],
        [sg.Text(list_text[12])],
        [sg.ProgressBar(len(image_filenames),
                        orientation='h',
                        size=(len(list_text[11] + str(root_dir)), 10),
                        border_width=1,
                        key='-PROG-')],
    ]
    # create the Window
    window_bar = sg.Window('', layout_bar, no_titlebar=False, finalize=True, disable_close=True)

    for file in image_filenames:
        slices.append(pydicom.dcmread(file))
        window_bar['-PROG-'].update(i + 1)
        i += 1
    window_bar.close()
    skip_count = 0
    for f in slices:
        if hasattr(f, 'SliceLocation'):
            new_slices.append(f)
        else:
            skip_count = skip_count + 1
        i += 1
    removed = []
    sh = slices[0].pixel_array.shape
    for s in new_slices:
        if sh != s.pixel_array.shape:
            removed.append(s)
    for r in removed:
        new_slices.remove(r)

    # ensure they are in the correct order
    # if new_slices[len(new_slices) // 2 + 1].SliceLocation > new_slices[len(new_slices) // 2 - 1].SliceLocation:
    if (new_slices[0].InstanceNumber < new_slices[-1].InstanceNumber) and (
            new_slices[0].SliceLocation < new_slices[-1].SliceLocation):
        new_slices = sorted(new_slices, key=lambda s: s.InstanceNumber)
        # new_slices = sorted(new_slices, key=lambda s: s.SliceLocation)
    else:
        new_slices = sorted(new_slices, reverse=True, key=lambda s: s.SliceLocation)
        # new_slices = sorted(new_slices, reverse = True, key=lambda s: s.InstanceNumber)

    return new_slices


def readDicomFiles(dicom_path):
    files = []
    slices = []
    SeriesDescription = []
    CountOfImages = []
    i = 0
    layout_bar = [
        [sg.Text(f'{list_text[11]} {dicom_path}')],
        [sg.Text(list_text[12])],
        [sg.ProgressBar(len(os.listdir(dicom_path)),
                        orientation='h',
                        size=(len(list_text[11] + str(dicom_path)), 10),
                        border_width=1,
                        key='-PROG-')],
    ]
    # create the Window
    window_bar = sg.Window('', layout_bar, no_titlebar=False, finalize=True, disable_close=True)

    # read dicom dataset from dir
    for s in os.listdir(dicom_path):
        slices.append(pydicom.dcmread(dicom_path + '/' + s))
        window_bar['-PROG-'].update(i + 1)
        i += 1

    # km.global_messages.append(f'Total files in dataset: {count_of_files}\n')
    window_bar.close()

    slices = sorted(slices, key=lambda sl: sl.SeriesNumber)

    series_numbers = [slices[0].SeriesNumber]
    SeriesDescription.append(slices[0].SeriesDescription)
    i = 0
    coi = 0
    for slice in slices:
        # print(series_numbers[i], slice.SeriesNumber)
        if series_numbers[i] == slice.SeriesNumber:
            coi += 1
        else:
            series_numbers.append(slice.SeriesNumber)  # get all series in slices
            SeriesDescription.append(slice.SeriesDescription)
            CountOfImages.append(coi)
            i += 1
            coi = 1
    CountOfImages.append(coi)

    selectedSerie = selectSeries(series_numbers, SeriesDescription, CountOfImages)
    # skip files with no SliceLocation (e.g. scout views)
    skip_count = 0
    removed = []
    i = 0
    new_slices = []

    ss = [ss for ss in slices if ss.SeriesNumber == selectedSerie]
    for f in ss:
        if hasattr(f, 'SliceLocation'):
            new_slices.append(f)
        else:
            skip_count = skip_count + 1
        i += 1

    sh = slices[0].pixel_array.shape
    for s in new_slices:
        if sh != s.pixel_array.shape:
            removed.append(s)
    for r in removed:
        new_slices.remove(r)

    # ensure they are in the correct order
    # if new_slices[len(new_slices) // 2 + 1].SliceLocation > new_slices[len(new_slices) // 2 - 1].SliceLocation:
    if (new_slices[0].InstanceNumber < new_slices[-1].InstanceNumber) and (new_slices[0].SliceLocation < new_slices[-1].SliceLocation):
        new_slices = sorted(new_slices, key=lambda s: s.InstanceNumber)
        # new_slices = sorted(new_slices, key=lambda s: s.SliceLocation)
    else:
        new_slices = sorted(new_slices, reverse=True, key=lambda s: s.SliceLocation)
        # new_slices = sorted(new_slices, reverse = True, key=lambda s: s.InstanceNumber)

    return new_slices


def read_dicom_set(dicom_path):
    global frameSize
    km.global_messages.clear()
    path = dicom_path + '/'
    dicomdirpath = (path + 'DICOMDIR') if os.path.isfile(path + 'DICOMDIR') else None
    if dicomdirpath is not None:
        # print(dicomdirpath)
        slices = readDICOMDIR(dicomdirpath)
    else:
        slices = readDicomFiles(path)

    # km.global_messages.append(f'Total skipped files, have no SliceLocation: {skip_count}\n')
    km.global_messages.append(f'Изображения, полученные по результатам компьютерной томографии\n')
    km.global_messages.append(f'Пациент: {slices[0].PatientName}, код пациента {slices[0].PatientID}\n')
    km.global_messages.append(f'Дата проведения исследования КТ: {slices[0].StudyDate}\n')
    # km.global_messages.append(
    #    f'All files in dir {dicom_path} of patient {slices[0].PatientName} includes series {series_numbers}\n')

    # Calculate frame size for image
    Length_image = slices[0].Rows * slices[0].PixelSpacing[0]
    Height_image = abs((slices[0].SliceLocation - slices[-1].SliceLocation))  # * len(new_slices)
    frameSize = (slices[0].Rows, int(slices[0].Rows * (Height_image / Length_image)))
    return slices, frameSize


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int32)
    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int32)
    image += np.int32(intercept)

    return np.array(image, dtype=np.int32)


def map2win(image_arr, window_level=-450, window_width=1500):
    """
     The purpose is to map the pixel values of the CT image (usually in a wide range, -2048~2048) to a fixed
            Within the scope, the mapping function needs to be calculated in conjunction with the window width window.
            Window window width of lung parenchyma
     window_level = -450~-600
     window_width = 1500~2000
    """
    window_max = window_level + 0.5 * window_width
    window_min = window_level - 0.5 * window_width
    index_min = image_arr < window_min
    index_max = image_arr > window_max
    #     index_mid = np.where((image >= window_min)&(image <= window_max))
    image_arr = (image_arr - window_min) / (window_width / 256) - 1
    image_arr[index_min] = 0
    image_arr[index_max] = 255
    return image_arr


def save_slice_to_image(image_name, slice_array, window_level, window_width,
                        frame_Size):  # improvement of image and save to dir
    slice_array = map2win(slice_array, window_level, window_width)
    x_size, y_size = frame_Size
    if x_size < 200:
        x_size = 200
    if y_size < 200:
        y_size = 200
    frame_Size = (x_size,y_size)
    slice_array = cv2.resize(slice_array.astype(np.int16), frame_Size, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(image_name, slice_array)
    return


def get_images_from_slice(data_array, images_path, slices, step=1):

    start_pos=(slices[0].Columns//2) - int(img_count)//2
    stop_pos=(slices[0].Columns//2) + int(img_count)//2
    # start_pos, stop_pos = get_par_of_image(imgs, datapath, len_data, rows, columns)
    # #get start and stop points for slicing

    n = int((stop_pos - start_pos) / step)

    layout_bar = [
        [sg.Text(f'{list_text[13]} {images_path}')],
        [sg.Text(list_text[12])],
        [sg.ProgressBar(n, orientation='h', size=(len(list_text[13] + images_path), 10), border_width=1, key='-PROG-')],
    ]
    # create the Window
    window_bar = sg.Window('', layout_bar, no_titlebar=False, finalize=True, disable_close=True)

    # get images and store to output directory
    count = 0
    dir_info = os.path.join(images_path + '/' + str(now.strftime("%d%m%y")) + 'dirinfo.txt')
    with open(dir_info, 'w') as f:
        f.write(f'Изображения, полученные по результатам компьютерной томографии\n')
        f.write(f'Пациент: {slices[0].PatientName}, код пациента {slices[0].PatientID}\n')
        f.write(f'Дата проведения исследования КТ: {slices[0].StudyDate}\n')
        f.write(f'Дата получения изображений из КТ: {now.strftime("%d-%m-%Y %H:%M")}\n')
        f.write(f'Размер изображения: {frameSize} \n')
        f.write(f'Список файлов изображений из КТ:\n')
        for y in range(start_pos, stop_pos):
            # filename of image
            out_image_name = os.path.join(
                images_path + '/' + str(slices[0].PatientID)[:8] + '_' + str(now.strftime("%d%m%y")) + '_' + str(
                    y) + '.' + str(img_format))
            out_image_name = out_image_name.strip(' ')
            coronal_slice = data_array[:, y, :]
            # coronal_slice = setDicomWinWidthWinCenter(coronal_slice, 400, 40, slices[0].Rows, slices[0].Columns)
            save_slice_to_image(out_image_name, coronal_slice, slices[0].WindowCenter, slices[0].WindowWidth, frameSize)
            count += 1
            window_bar['-PROG-'].update(count)
            f.write(f'{out_image_name}\n')
        f.write(f'Всего сформировано файлов изображений из КТ: {count}\n')
        window_bar.close()

    km.global_messages.append(f'Total wrote {count} images into directory {images_path}\n')


def save_array(array, save_dir, slices):
    # save numpy array from dicom
    global patient_ID
    patient_ID = slices[0].PatientID[:8]
    numpy_array_name = save_dir + '/' + patient_ID + 'array.npy'
    np.save(numpy_array_name, array)
    # np.save(numpy_array_name, array if (slices[0].SliceLocation - slices[-1].SliceLocation < 0) else array[:, ::-1,
    # :])
    array_info = save_dir + '/' + patient_ID + 'arrayinfo.csv'
    array_info_txt = save_dir + '/' + patient_ID + 'arrayinfo.txt'
    with open(array_info, mode="w", encoding='utf-8', newline="") as w_file:
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
                      "X",
                      "Window Center",
                      "Window Width"
                      ]
        file_writer = csv.DictWriter(w_file, fieldnames=fieldnames, delimiter=',')
        file_writer.writeheader()
        file_writer.writerow({
            "Study Date": slices[0].StudyDate,
            "Series Description": slices[0].SeriesDescription,
            "Patient's Name": slices[0].PatientName,
            "Patient ID": slices[0].PatientID,
            "Spacing Between Slices": slices[0].SpacingBetweenSlices,
            "Series Number": slices[0].SeriesNumber,
            "Start Slice Location": slices[0].SliceLocation,
            "End Slice Location": slices[-1].SliceLocation,
            "Slice Thickness": slices[0].SliceThickness,
            "Rows": slices[0].Rows,
            "Columns": slices[0].Columns,
            "Samples per Pixel": slices[0].SamplesPerPixel,
            "Pixel Spacing X": slices[0].PixelSpacing[0],
            "Pixel Spacing Y": slices[0].PixelSpacing[1],
            "Rescale Intercept": slices[0].RescaleIntercept,
            "Rescale Slope": slices[0].RescaleSlope,
            "Shape Z, Y, X": array.shape,
            "Z": array.shape[0],
            "Y": array.shape[1],
            "X": array.shape[2],
            "Window Center": slices[0].WindowCenter,
            "Window Width": slices[0].WindowWidth
        })

    with open(array_info_txt, 'w') as f:
        f.write(f'Study Date: {slices[0].StudyDate}\n')
        f.write(f'Series Description: {slices[0].SeriesDescription}\n')
        f.write(f"Patient's Name: {slices[0].PatientName}\n")
        f.write(f"Patient ID: {slices[0].PatientID}\n")
        f.write(f"Spacing Between Slices: {slices[0].SpacingBetweenSlices}\n")
        f.write(f"Series Number: {slices[0].SeriesNumber}\n")
        f.write(f"Start Slice Location: {slices[0].SliceLocation}\n")
        f.write(f"End Slice Location: {slices[-1].SliceLocation}\n")
        f.write(f"Slice Thickness: {slices[0].SliceThickness}\n")
        f.write(f"Rows: {slices[0].Rows}\n")
        f.write(f"Columns: {slices[0].Columns}\n")
        f.write(f"Samples per Pixel: {slices[0].SamplesPerPixel}\n")
        f.write(f"Pixel Spacing X: {slices[0].PixelSpacing[0]}\n")
        f.write(f"Pixel Spacing Y: {slices[0].PixelSpacing[1]}\n")
        f.write(f"Rescale Intercept: {slices[0].RescaleIntercept}\n")
        f.write(f"Rescale Slope: {slices[0].RescaleSlope}\n")
        f.write(f"Shape of array Z, Y, X: {array.shape}\n")
        f.write(f"Window Center: {slices[0].WindowCenter}\n")
        f.write(f"Window Width: {slices[0].WindowWidth}\n")

    km.global_messages.append(f'Data array saved to file {numpy_array_name}, volume of array: {array.shape}\n')


def rem_spase(s):
    s = s.strip().split(" ")
    s = "_".join(s)
    s = s.strip().split("^")
    s = "_".join(s)
    return s


def change_slash(s):
    return '/'.join(s.strip().split('\\'))


def readDicomFolder(dicom_path, images_path):
    # get slice from dicom dataset
    slices, frameSize = read_dicom_set(dicom_path)
    pixel_array = get_pixels_hu(slices)
    # set path to save images
    patID = str(slices[0].PatientID)[:8]
    patName = str(slices[0].PatientName).strip('^')
    save_dir = os.path.join(images_path, patID + '_' + rem_spase(patName))
    save_dir = save_dir.strip('^')
    save_dir = save_dir + '/' + str(now.strftime("%d%m%y"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        lay_mess = [
            [sg.Text(f'{list_text[14]} {save_dir}')]
        ]
        warning = sg.Window(window_heads[1], lay_mess, auto_close=True, auto_close_duration=5, finalize=True)
        event, values = warning.read()
        if event == sg.WIN_CLOSED:
            warning.close()
    # get images from array and store to output directory
    get_images_from_slice(pixel_array, save_dir, slices, step=1)
    save_array(pixel_array, save_dir, slices)
    layout_messages = [
        [sg.Text('\n'.join(map(str, km.global_messages)))],
        [sg.Button(button_text=buttons[1], size=(len(buttons[1]), 2)),
         sg.OK(button_text=buttons[5], size=(10, 2), button_color='teal')]
    ]
    window_message = sg.Window(window_heads[3], layout_messages, default_element_size=(60, 2))
    event, values = window_message.read()
    if event in (sg.WIN_CLOSED, buttons[5]):
        window_message.close()
        km.global_messages.clear()
    elif event == buttons[1]:
        window_message.close()
        km.view_image_folder(save_dir)


def main(dicom_path, images_path):
    readDicomFolder(dicom_path, images_path)
    return


'''
if __name__ == "__main__":
    init_folder = km.default_input_dicom_folder
    start_folder = km.default_output_folder
    data_path = get_dicom_path(init_folder)
    output_path = get_images_path(start_folder)
    main(data_path, output_path)
'''
