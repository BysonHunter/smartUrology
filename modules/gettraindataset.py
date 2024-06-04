import PySimpleGUI as sg
from pathlib import Path
import os
import shutil
from PIL import Image
import cv2

from modules import getimagefromdicom as gi
from modules import config as cf

lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme, \
    img_count, img_format = cf.read_settings()
tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)


# noinspection PyBroadException
def read_label_file(filename):
    text = ''
    if Path(filename).is_file():
        try:
            with open(filename, "rt") as f:
                text = f.read()
        except Exception:
            text = list_text[3]
    return text


def edit_label_file(filename, text):
    if Path(filename).is_file():
        with open(filename, "w") as f:
            f.write(text)
    return


def copy_to_dataset(filename, output_path):
    n = len(filename)
    count = 0
    layout_bar = [
        [sg.Text(f'{list_text[13]} {output_path}')],
        [sg.Text(list_text[12])], [sg.Text(f'copied {count} files', key='-count-')],
        [sg.ProgressBar(n, orientation='h', size=(50, 10), border_width=1, key='-PROG-')],
    ]
    # create the Window
    window_bar = sg.Window('', layout_bar, no_titlebar=False, finalize=True)

    for f in filename:
        label_filename = str(f)[:-20] + '/labels' + str(f)[-20:-4] + '.txt'
        source = str(Path(f).parent.parent / Path(f).name)
        destination = str(output_path) + '/images/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.copy(source, destination)
        source = label_filename
        destination = str(output_path) + '/labels/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.copy(source, destination)
        window_bar['-PROG-'].update(count + 1)
        window_bar['-count-'].update(count)
    window_bar.close()


def delete_files(filename):
    image_file = filename
    label_file = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
    os.remove(image_file)
    os.remove(label_file)


def save_annotations(filename, object_class, img, bboxes):
    """
    Saves annotations to a text file in YOLO format,
    class, x_centre, y_centre, width, height
    """
    img_height = img.shape[0]
    img_width = img.shape[1]

    with open(filename, 'a') as f:
        for bbox in bboxes:
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]

            x1 = int(bbox[0])  # - int(int(r[2])/2)
            x2 = int(bbox[0]) + int(int(bbox[2]))
            y1 = int(bbox[1])  # - int(int(r[3])/2)
            y2 = int(bbox[1]) + int(int(bbox[3]))

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            width = x2 - x1
            height = y2 - y1

            x_centre = int(width / 2)
            y_centre = int(height / 2)

            norm_xc = x_centre / img_width
            norm_yc = y_centre / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            '''
            yolo_annotations = [str(object_class), ' ' + str(norm_xc),
                                ' ' + str(norm_yc),
                                ' ' + str(norm_width),
                                ' ' + str(norm_height), '\n']

            '''
            yolo_annotations = [
                f'{object_class} {norm_xc :.7f} {norm_yc :.7f} {norm_width :.7f} {norm_height :.7f}\n']
            f.writelines(yolo_annotations)


def select_class_objectSeries():
    chance = ['Kевая почка правильной формы – «left_kidney»',
              'Конкремент (камень) правильной формы – «stone»',
              'Правая почка правильной формы – «right_kidney»',
              'Патологически увеличенная левая почка – «left_kidney_pieloectasy»',
              'Патологически увеличенная правая почка – «right_kidney_pieloectasy»',
              'Камень коралловидной формы - класс «staghorn_stones»'
              ]

    class_select_column = [
        [sg.Listbox(
            values=chance,
            enable_events=True,
            size=(50, len(chance)),
            key='-OBJ CLASS LIST-',
        )],
        [sg.OK('Выбрать', key='OK')]
    ]
    selectedSerieNumber = chance[0]
    index_of_class = 0

    window_select_class = sg.Window('Выбор класса объекта', layout=class_select_column,
                                    # size=screensize,
                                    return_keyboard_events=True,
                                    finalize=True,
                                    location=(0, 0),
                                    resizable=False,
                                    no_titlebar=False)

    window_select_class['-OBJ CLASS LIST-'].update(chance, set_to_index=index_of_class, scroll_to_index=index_of_class)

    while True:
        event, values = window_select_class.read()

        if event in ('MouseWheel:Down', 'Down:40', 'Next:34') and index_of_class < len(chance) - 1:
            index_of_class += 1
            window_select_class['-OBJ CLASS LIST-'].update(set_to_index=index_of_class, scroll_to_index=index_of_class)

        elif event in ('MouseWheel:Up', 'Up:38', 'Prior:33') and index_of_class > 0:
            index_of_class -= 1
            window_select_class['-OBJ CLASS LIST-'].update(set_to_index=index_of_class, scroll_to_index=index_of_class)

        elif event in (sg.WIN_CLOSED, 'OK', 'Escape:27'):
            selectedSerieNumber = index_of_class
            window_select_class.close()
            break

        window_select_class['-OBJ CLASS LIST-'].update(chance, set_to_index=index_of_class,
                                                       scroll_to_index=index_of_class)

    return selectedSerieNumber


def mark_object(cls_of_obj, ct_img, label_filename):
    class_object = cls_of_obj
    showCrosshair = False
    fromCenter = False
    rects = []
    im = cv2.imread(ct_img)

    img_height = im.shape[0]
    img_width = im.shape[1]

    bbox = []
    boxes = []

    r = cv2.selectROI("Image", im, fromCenter)

    bbox.clear()
    bbox.append(int(r[0]))
    bbox.append(int(r[1]))
    bbox.append(int(r[2]))
    bbox.append(int(r[3]))
    boxes.append(bbox)

    x1 = int(r[0])  # - int(int(r[2])/2)
    x2 = int(r[0]) + int(int(r[2]))
    y1 = int(r[1])  # - int(int(r[3])/2)
    y2 = int(r[1]) + int(int(r[3]))

    rec_colors = [
        (57, 73, 171),
        (3, 155, 229),
        (0, 172, 193),
        (0, 137, 123),
        (67, 160, 71),
        (124, 179, 66),
    ]
    color = rec_colors[cls_of_obj]

    save_annotations(label_filename, class_object, im, boxes)
    cv2.rectangle(im, (x1, y1), (x2, y2), color, thickness=2)
    cv2.imwrite(ct_img, im)

    cv2.waitKey(0)


def get_train_dataset(input_path, output_path, frameSize=(128, 128)):
    folder = input_path + '/detect/'
    # if select error folder, not have label files 
    if not os.path.exists(os.path.join(folder + '/labels')):
        lay_mess = [
            [sg.Text(f'{folder} {list_text[4]}',
                     justification='center',
                     text_color='white',
                     background_color='brown'), ]
        ]
        warning = sg.Window(window_heads[1], lay_mess,
                            size=(800, 80),
                            titlebar_background_color='brown',
                            auto_close=True,
                            background_color='brown',
                            auto_close_duration=3,
                            finalize=True)

        event, values = warning.read()
        if event == sg.WIN_CLOSED:
            warning.close()
        return

    # png_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(img_format)]
    # filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(img_format)]
    # filenum, filename = 0, png_files[0]

    images_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
    # print(images_filenames)
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
    filenum, filename = 0, images_filenames[0]
    labeltext = ''
    files_to_dataset = []

    chance = ['Левая почка правильной формы – «left_kidney»',
              'Конкремент (камень) правильной формы – «stone»',
              'Правая почка правильной формы – «right_kidney»',
              'Патологически увеличенная левая почка – «left_kidney_pieloectasy»',
              'Патологически увеличенная правая почка – «right_kidney_pieloectasy»',
              'Камень коралловидной формы - класс «staghorn_stones»'
              ]

    label_view = [
        [sg.Text(list_text[5], justification='center'), ],
        [sg.Text('', size=(51, 3), key="-LABEL NAME-"), ],
        [sg.Multiline(default_text=labeltext, size=(50, 10),
                      enable_events=True, key="-LABEL-")],
    ]

    file_list_column = [
        [sg.Text(list_text[0])],
        [sg.In(size=(45, 1), enable_events=True, key="-FOLDER-"),
         sg.FolderBrowse(button_text=buttons[13],
                         initial_folder=folder, tooltip=folder, size=(10, 2)),
         ],
        [sg.Listbox(values=filenames_only, horizontal_scroll=True,
                    enable_events=True, size=(25, 20), key="-FILE LIST-")],
        # [sg.HSeparator()],
        [sg.Text(list_text[1], justification='center', size=(45, 1))],
        [sg.HSeparator()],
        [sg.Checkbox(checkbox[0], enable_events=True,
                     text_color='coral', background_color='lavender',
                     tooltip=tooltips[6],
                     size=(45, 1), key="-SELECT_ALL-")],
        [sg.Button(button_text=buttons[12], size=(47, 1), button_color='green'), ],
        [sg.HSeparator()],
        [sg.Cancel(button_text=buttons[5], size=(47, 1), button_color='teal'), ],
    ]
    image_viewer_column = [
        [sg.Text(list_text[2])],
        [sg.Text(size=(60, 1), key="-TOUT-")],
        [sg.Image(size=(int(frameSize[0] * 0.75), int(frameSize[1] * 0.75)), key="-IMAGE-")],
        [sg.Text('File 1 of {}'.format(len(filenames_only)), size=(15, 1), key='-FILENUM-')],
        [sg.Button(buttons[6], size=(10, 1)),
         sg.Button(buttons[7], size=(10, 1)),
         sg.Button(buttons[8], size=(10, 1)),
         sg.Button(buttons[9], size=(10, 1))],
        [sg.Button('Добавить объект', size=(47, 1), key='-ADD_OBJ-')],
        [sg.Button('Удалить объект', size=(47, 1), key='-DEL_OBJ-')],
        [sg.Button(buttons[11], size=(47, 1), button_color='red', highlight_colors=('green', 'white'),
                   tooltip=tooltips[7])],
    ]

    class_select_column = [
        [sg.Text('Выбрать класс объекта', justification='center', size=(45, 1))],
        [sg.Listbox(
            values=chance,
            enable_events=True,
            size=(50, len(chance)),
            key='-OBJ CLASS LIST-',
        )],
        # [sg.OK('Выбрать', key='OK')]
    ]

    layout = [
        [sg.Column(file_list_column),
         sg.VSeperator(),
         sg.Column(image_viewer_column),
         # sg.VSeperator(),
         sg.Column(label_view),
         sg.Column(class_select_column),
         ],
    ]
    # win_h, win_w = screensize
    window_view = sg.Window(window_heads[2], layout, use_default_focus=True,
                            # size=(int(win_h*0.75), int(win_w*0.85)),
                            # element_padding=2,
                            return_keyboard_events=True, finalize=True,
                            location=(0, 0),
                            resizable=True, modal=True,
                            no_titlebar=False)

    label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'

    # label_objects = []
    text = read_label_file(label_filename)
    # label_objects.append(text)
    # print(label_objects)

    select_all = False
    cls_of_obj = 0
    cls_of_obj_new = 0

    view_img_filename = os.path.join(input_path, filenames_only[0][:-4] + '.png')
    if img_format == 'jpg':
        image = Image.open(view_img_filename[:-4] + '.jpg')
        image.save(view_img_filename[:-4] + '.png', format="PNG")

    window_view["-LABEL NAME-"].update(label_filename)
    window_view["-LABEL-"].update(text)
    window_view["-FOLDER-"].update(input_path)
    window_view["-FILE LIST-"].update(filenames_only, set_to_index=filenum, scroll_to_index=filenum)
    window_view["-TOUT-"].update(os.path.join(input_path, filenames_only[0]))
    window_view["-IMAGE-"].update(filename=view_img_filename)
    window_view["-OBJ CLASS LIST-"].update(chance, set_to_index=cls_of_obj, scroll_to_index=cls_of_obj)

    if img_format == 'jpg':
        os.remove(view_img_filename[:-4] + '.png')

    while True:
        event, values = window_view.read()
        if event == "-FOLDER-":
            folder = values["-FOLDER-"] + '/detect/'
            filenum = 0
            png_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(img_format)]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(img_format)]
            filename = os.path.join(folder, filenames_only[filenum])

            window_view["-FOLDER-"].update(folder)
            window_view["-FILE LIST-"].update(filenames_only)
            window_view["-TOUT-"].update(os.path.join(folder, filenames_only[0]))
            view_image_name = os.path.join(folder, filenames_only[0])
            if img_format == 'jpg':
                image = Image.open(view_image_name[:-4] + '.jpg')
                image.save(view_image_name[:-4] + '.png', format="PNG")
            window_view["-IMAGE-"].update(filename=view_image_name[:-4] + '.png')
            if img_format == 'jpg':
                os.remove(view_image_name[:-4] + '.png')
            # window_view["-IMAGE-"].update(filename=os.path.join(folder, filenames_only[0]))

        elif event in (buttons[8], 'MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(filenames_only) - 1:
            filenum += 1
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels/' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
            window_view["-LABEL NAME-"].update(label_filename)
            window_view["-LABEL-"].update(text)
        elif event in (buttons[7], 'MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
            filenum -= 1
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels/' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
            window_view["-LABEL NAME-"].update(label_filename)
            window_view["-LABEL-"].update(text)
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
        elif event == buttons[6]:
            filenum = 0
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
        elif event == buttons[9]:
            filenum = len(filenames_only) - 1
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            filename = os.path.join(folder, values['-FILE LIST-'][0])
            filenum = png_files.index(filename)
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
        elif event == buttons[12]:
            if not select_all:
                files_to_dataset.clear()
                files_to_dataset.append(filename)
            else:
                files_to_dataset.clear()
                files_to_dataset = png_files
            copy_to_dataset(files_to_dataset, output_path)
        elif event == "-SELECT_ALL-":
            if not select_all:
                select_all = True
            else:
                select_all = False
        elif event in (buttons[11], 'Delete:46'):
            delete_files(filename)
            # filenum = 0
            if filenum >= len(filenames_only) - 1:
                filenum -= 1
            png_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(img_format)]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(img_format)]
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
            # window_view["-FILE LIST-"].update(set_to_index=filenum+1, scroll_to_index=filenum+1)  

        elif event == 'Edit label file...':
            edit_label_file(filename, text)

        elif event == "-OBJ CLASS LIST-":
            cls_of_obj_new = chance.index(values["-OBJ CLASS LIST-"][cls_of_obj])
            # cls_of_obj = cls_of_obj_new

        elif event == '-ADD_OBJ-':
            image_name = filename
            label_name = label_filename
            mark_object(cls_of_obj_new, image_name, label_name)
            window_view["-LABEL NAME-"].update(label_name)
            # print(label_name, image_name)

        elif event == '-DEL_OBJ-':
            label_objects = []
            labels_for_delete = []
            label_objects.clear()
            text_new = read_label_file(label_filename)
            # label_objects.append(text_new)
            for line in text_new.split("\n"):
                if not line.strip():
                    continue
                labels_for_delete.append(line.lstrip())

            with open(filename, "w") as f:
                for lbl_for_del in labels_for_delete:
                    if labels_for_delete[0][0] != cls_of_obj_new:
                        f.writelines(lbl_for_del)

            window_view["-LABEL NAME-"].update(label_filename)

        elif event in (sg.WIN_CLOSED, buttons[5], 'Escape:27'):
            window_view.close()
            break

        window_view["-FILE LIST-"].update(filenames_only, set_to_index=filenum, scroll_to_index=filenum)
        window_view["-LABEL NAME-"].update(label_filename)
        window_view["-LABEL-"].update(text)
        window_view["-TOUT-"].update(filename)
        if img_format == 'jpg':
            # img = cv2.imread(filename[:-4] + '.jpg')
            # color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            open_image_name = filename[:-4] + '.jpg'
            # im_pil = Image.fromarray(img)
            image = Image.open(open_image_name)
            save_image_name = filename[:-4] + '.png'
            image.save(save_image_name, format="PNG")

        window_view["-IMAGE-"].update(filename=save_image_name)
        if img_format == 'jpg':
            os.remove(filename[:-4] + '.png')
        # window_view["-IMAGE-"].update(filename=filename)

        window_view['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(images_filenames)))
        window_view["-OBJ CLASS LIST-"].update(chance, set_to_index=cls_of_obj_new, scroll_to_index=cls_of_obj_new)


def main():
    input_path = gi.get_images_path(str(Path.cwd().parent / 'out'))
    output_path = gi.get_images_path(str(Path.cwd().parent / 'traindataset'))
    get_train_dataset(input_path, output_path)


if __name__ == "__main__":
    main()
