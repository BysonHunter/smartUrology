import os
import sys
from os import environ
from pathlib import Path
import PySimpleGUI as sg
from PIL import Image

from modules import calcstoneparam as sp
from modules import config as cf
from modules import detectstones as ds
from modules import getimagefromdicom as gi
from modules import gettraindataset as gs

# from tkinter import Tk
# root = Tk()
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

# get screen resolution and set windows size ------------------------------------------------------
from PyQt5.QtWidgets import QDesktopWidget, QApplication
app = QApplication(sys.argv)
screen_width = QDesktopWidget().availableGeometry().width()
screen_height = QDesktopWidget().availableGeometry().height()
# -------------------------------------------------------------------------------------------------
global_messages = []


def sets():
    global lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, \
        current_color_theme, img_count, img_format, tooltips, buttons, list_text, checkbox, main_menu, window_heads, \
        titles

    lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, current_color_theme, \
        img_count, img_format = cf.read_settings()

    sg.theme(current_color_theme)
    tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = cf.set_language(lang_of_interface)


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


def view_image_folder(input_path, frameSize=(512, 512)):
    sets()
    folder = input_path
    png_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png')]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    filenum, filename = 0, png_filenames[0]
    save_confidence = True

    file_list_column = [
        [sg.Text(list_text[0])],
        [sg.In(size=(42, 1), enable_events=True, key="-FOLDER-"),
         sg.FolderBrowse(button_text=buttons[13], initial_folder=folder, tooltip=folder, size=(11, 1)),
         ],
        [sg.Listbox(
            values=filenames_only, horizontal_scroll=True, enable_events=True, size=(50, 30), key="-FILE LIST-"
        )],
        [sg.HSeparator()],
        [sg.Text(list_text[1], justification='center', size=(45, 2))],
        [sg.HSeparator()],
        [sg.Radio(buttons[17], "RADIO1", default=True, enable_events=True, key="-select-model1-"),
         sg.Radio(buttons[18], "RADIO1", default=False, enable_events=True, key="-select-model2-", visible=False)
         ],
        [sg.Checkbox(checkbox[1], enable_events=True,
                     default=True,
                     text_color='coral', background_color='lavender',
                     tooltip=tooltips[6],
                     size=(45, 1),
                     key="-save_confidence-")],

        [sg.Button(button_text=buttons[10], size=(22, 2)),
         sg.Cancel(button_text=buttons[5], size=(22, 2), button_color='teal'), ],
    ]

    image_viewer_column = [
        [sg.Text(list_text[2])],
        [sg.Text(size=(62, 1), key="-TOUT-")],
        [sg.Image(size=frameSize, key="-IMAGE-")],
        [
            sg.Button(button_text=buttons[6], size=(8, 2)),
            sg.Button(button_text=buttons[7], size=(8, 2)),
            sg.Button(button_text=buttons[8], size=(8, 2)),
            sg.Button(button_text=buttons[9], size=(8, 2)),
            sg.Text('File 1 of {}'.format(len(filenames_only)), size=(15, 1), key='-FILENUM-')]
    ]

    layout = [
        [sg.Column(file_list_column, vertical_alignment='top'),
         # sg.VSeperator(),
         sg.Column(image_viewer_column, vertical_alignment='top'),
         ],
    ]
    # win_h, win_w = screensize
    window_view = sg.Window(window_heads[2], layout, use_default_focus=True,
                            # size=(int(win_h*0.75), int(win_w*0.85)),
                            # size=(1024, 768),
                            # element_padding=2,
                            # size=screensize,
                            size=(QDesktopWidget().availableGeometry().width(),
                                  QDesktopWidget().availableGeometry().height()),
                            return_keyboard_events=True, finalize=True,
                            location=(0, 0), resizable=True,
                            no_titlebar=False)

    window_view["-FOLDER-"].update(input_path)
    window_view["-FILE LIST-"].update(filenames_only)
    window_view["-TOUT-"].update(os.path.join(input_path, filenames_only[0]))
    window_view["-IMAGE-"].update(filename=os.path.join(input_path, filenames_only[0]))
    window_view.maximize()

    while True:
        event, values = window_view.read()
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            filenum = 0
            png_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.png')]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith('.png')]
            filename = os.path.join(folder, filenames_only[filenum])
            window_view["-FOLDER-"].update(folder)
            window_view["-FILE LIST-"].update(filenames_only)
            window_view["-TOUT-"].update(os.path.join(folder, filenames_only[0]))
            window_view["-IMAGE-"].update(filename=os.path.join(folder, filenames_only[0]))
        elif event in (buttons[8], 'MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(filenames_only) - 1:
            filenum += 1
            filename = os.path.join(folder, filenames_only[filenum])
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
        elif event in (buttons[7], 'MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
            filenum -= 1
            filename = os.path.join(folder, filenames_only[filenum])
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
        elif event == buttons[6]:
            filenum = 0
            filename = os.path.join(folder, filenames_only[filenum])
        elif event == buttons[9]:
            filenum = len(filenames_only) - 1
            filename = os.path.join(folder, filenames_only[filenum])
        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            filename = os.path.join(folder, values['-FILE LIST-'][0])
            filenum = png_filenames.index(filename)
        elif event == buttons[10]:
            window_view.close()
            global_messages.clear()
            if values["-select-model1-"]:
                ds.detect_stones(folder, save_confidence, path_to_yolo_weights)
                view_detected_images(folder + '/detect/')
            if values["-select-model2-"]:
                # dsn.detect_stones(folder, save_confidence)
                # view_detected_images(folder + '/detect_new/')
                pass
            break
        elif event == "-save_confidence-":
            if not save_confidence:
                save_confidence = True
            else:
                save_confidence = False
        elif event in (sg.WIN_CLOSED, buttons[5], 'Escape:27'):
            window_view.close()
            break
        window_view["-TOUT-"].update(filename)
        window_view["-IMAGE-"].update(filename=filename)
        window_view['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(png_filenames)))


def view_detected_images(input_path, frameSize=(512, 512)):
    # noinspection PyBroadException
    def read_label_file(file_name):
        text_label = ''
        if Path(file_name).is_file():
            try:
                with open(file_name, "rt") as f:
                    text_label = f.read()
            except Exception:
                text_label = list_text[3]
        return text_label

    folder = input_path

    # if select error folder, not have label files
    if not os.path.exists(os.path.join(folder + '/labels')) and folder != '':
        lay_mess = [
            [sg.Text(f'{folder} {list_text[4]}',
                     text_color='white', background_color='brown'), ]
        ]
        warning_window = sg.Window(window_heads[1], lay_mess,
                                   auto_close=True,
                                   background_color='brown',
                                   auto_close_duration=5,
                                   finalize=True)
        event, values = warning_window.read()
        if event == sg.WIN_CLOSED:
            warning_window.close()
        return

    labeltext = ''

    label_view = [
        [sg.Text(list_text[5], justification='center')],
        [sg.Text('', size=(52, 3), key="-LABEL NAME-")],
        [sg.Multiline(default_text=labeltext, size=(52, 5),
                      enable_events=True, key="-LABEL-",
                      auto_size_text=True),
         ],
    ]

    img_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
    filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
    if len(img_filenames) > 0:
        filenum = 0
        filename = img_filenames[0]
        label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
    # label_filename = [f for f in os.listdir(os.path.join(folder, '/labels')) if f.lower().endswith('.txt')]
    else:
        sg.popup_error(list_text[10] + list_text[6][16:], title=titles[2],
                       auto_close=True,
                       auto_close_duration=3,
                       background_color='brown')
        return

    file_list_column = [
        [sg.Text(list_text[6])],
        [sg.In(size=(42, 1), enable_events=True, key="-FOLDER-"),
         sg.FolderBrowse(button_text=buttons[13],
                         initial_folder=folder, tooltip=folder, size=(11, 1)),
         ],
        [sg.Listbox(
            values=filenames_only, horizontal_scroll=True, enable_events=True, size=(50, 30), key="-FILE LIST-",
            auto_size_text=True)
        ],
        [sg.HSeparator()],
        [sg.Text(list_text[1], justification='center', size=(45, 2))],
        [sg.HSeparator()],
        [sg.Button(button_text=buttons[19], size=(22, 2))],
        [sg.Cancel(button_text=buttons[5], size=(22, 1), button_color='teal')],
    ]

    image_viewer_column = [
        [sg.Text(list_text[2])],
        [sg.Text(size=(65, 1), key="-TOUT-")],
        [sg.Image(size=frameSize, key="-IMAGE-")],
        [sg.Column(label_view)],
        [
            sg.Button(button_text=buttons[6], size=(8, 2)),
            sg.Button(button_text=buttons[7], size=(8, 2)),
            sg.Button(button_text=buttons[8], size=(8, 2)),
            sg.Button(button_text=buttons[9], size=(8, 2)),
            sg.Text('File 1 of {}'.format(len(filenames_only)), size=(32, 1), key='-FILENUM-')],
        [sg.Button(buttons[11], size=(47, 1), button_color='red', highlight_colors=('green', 'white'),
                   tooltip=tooltips[7])],
    ]

    layout = [
        [sg.Column(file_list_column, vertical_alignment='top'),
         # sg.VSeperator(),
         sg.Column(image_viewer_column, vertical_alignment='top'),
         # sg.VSeperator(),
         # sg.Column(label_view),
         ],
    ]

    text = read_label_file(label_filename)

    window_view = sg.Window(window_heads[2], layout,
                            use_default_focus=True,
                            # size=(int(win_h*0.75), int(win_w*0.85)),
                            # element_padding=2,
                            return_keyboard_events=True, finalize=True, resizable=True,
                            location=(0, 0),
                            size=(screen_width, screen_height),
                            no_titlebar=False)

    window_view["-LABEL NAME-"].update(label_filename)
    window_view["-LABEL-"].update(text)

    view_img_filename = os.path.join(input_path, filenames_only[0][:-4]+'.png')
    if img_format == 'jpg':
        image = Image.open(view_img_filename[:-4]+'.jpg')
        image.save(view_img_filename[:-4]+'.png', format="PNG")
    window_view["-FOLDER-"].update(input_path)
    window_view["-FILE LIST-"].update(filenames_only)
    window_view["-TOUT-"].update(os.path.join(input_path, filenames_only[0]))
    window_view["-IMAGE-"].update(filename=view_img_filename[:-4]+'.png')
    window_view.maximize()
    if img_format == 'jpg':
        os.remove(view_img_filename[:-4] + '.png')
    '''   
    window_view["-FOLDER-"].update(input_path)
    window_view["-FILE LIST-"].update(filenames_only)
    window_view["-TOUT-"].update(os.path.join(input_path, filenames_only[0]))
    window_view["-IMAGE-"].update(filename=os.path.join(input_path, filenames_only[0]))
    window_view.maximize()
    '''
    while True:
        event, values = window_view.read()
        if event == "-FOLDER-":
            folder = values["-FOLDER-"] # + '/detect/'
            filenum = 0
            img_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
            filename = os.path.join(folder, filenames_only[filenum])

            window_view["-FOLDER-"].update(folder)
            window_view["-FILE LIST-"].update(filenames_only)
            window_view["-TOUT-"].update(os.path.join(folder, filenames_only[0]))

            view_image_name = os.path.join(folder, filenames_only[0])
            if img_format == 'jpg':
                image = Image.open(view_image_name[:-4] + '.jpg')
                image.save(view_image_name[:-4] + '.png', format="PNG")
            window_view["-IMAGE-"].update(filename=view_image_name[:-4]+'.png')
            if img_format == 'jpg':
                os.remove(view_image_name[:-4] + '.png')

            # window_view["-IMAGE-"].update(filename=os.path.join(folder, filenames_only[0]))

        elif event in (buttons[8], 'MouseWheel:Down', 'Down:40', 'Next:34') and filenum < len(filenames_only) - 1:
            filenum += 1
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
            window_view["-LABEL NAME-"].update(label_filename)
            window_view["-LABEL-"].update(text)

        elif event in (buttons[7], 'MouseWheel:Up', 'Up:38', 'Prior:33') and filenum > 0:
            filenum -= 1
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)
            window_view["-FILE LIST-"].update(set_to_index=filenum, scroll_to_index=filenum)
            window_view["-LABEL NAME-"].update(label_filename)
            window_view["-LABEL-"].update(text)

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

        elif event in (buttons[11], 'Delete:46'):
            gs.delete_files(filename)
            # filenum = 0
            if filenum >= len(filenames_only) - 1:
                filenum -= 1
            img_filenames = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
            filenames_only = [f for f in os.listdir(folder) if f.lower().endswith(str(img_format))]
            filename = os.path.join(folder, filenames_only[filenum])
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)

        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            filename = os.path.join(folder, values['-FILE LIST-'][0])
            filenum = img_filenames.index(filename)
            label_filename = str(filename)[:-20] + '/labels' + str(filename)[-20:-4] + '.txt'
            text = read_label_file(label_filename)

        elif event == buttons[19]:
            sp.main(Path(folder).parent)

        elif event in (sg.WIN_CLOSED, buttons[5], 'Escape:27'):
            window_view.close()
            break

        window_view["-LABEL NAME-"].update(label_filename)
        window_view["-LABEL-"].update(text)
        window_view["-TOUT-"].update(filename)
        if img_format == 'jpg':
            image = Image.open(filename[:-4] + '.jpg')
            image.save(filename[:-4] + '.png', format="PNG")
        window_view["-IMAGE-"].update(filename=filename[:-4] + '.png')
        if img_format == 'jpg':
            os.remove(filename[:-4] + '.png')

        # window_view["-IMAGE-"].update(filename=filename)

        window_view['-FILENUM-'].update('File {} of {}'.format(filenum + 1, len(img_filenames)))


def warning(input_path, massage):
    lay_mess = [
        [sg.Text(f' {input_path} {massage}',
                 justification='center',
                 text_color='white',
                 background_color='brown'), ]
    ]
    warning_window = sg.Window(window_heads[1], lay_mess,
                               size=(800, 80),
                               titlebar_background_color='brown',
                               auto_close=True,
                               background_color='brown',
                               auto_close_duration=3,
                               finalize=True)

    event, values = warning_window.read()
    if event == sg.WIN_CLOSED:
        warning_window.close()
    return


def main():
    # main_menu = interface.main_menu
    global lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, \
        current_color_theme, img_count

    png_files = []
    save_confidence = True
    buttons_def = [
        [sg.Button(button_text=buttons[0], border_width=1, size=(45, 2), tooltip=tooltips[0])],
        [sg.Button(button_text=buttons[1], border_width=1, size=(45, 2), tooltip=tooltips[1])],
        # [sg.Button(button_text=buttons[2], border_width=1, size=(25, 2), tooltip=tooltips[2])],
        [sg.Button(button_text=buttons[3], border_width=1, size=(45, 2), tooltip=tooltips[3])],
        [sg.Button(button_text=buttons[4], border_width=1, size=(45, 2), tooltip=tooltips[4])],
        [sg.Button(button_text=buttons[14], border_width=1, size=(45, 2), tooltip=tooltips[9])],
        [sg.Cancel(button_text=buttons[5], size=(45, 2), button_color='teal', tooltip=tooltips[5])],
    ]

    main_window_layout = [
        [buttons_def],
    ]

    main_window = sg.Window(window_heads[0],
                            layout=main_window_layout,
                            # modal=True,
                            location=(0, 0),
                            margins=(0, 0),
                            size=(screen_width, screen_height),
                            auto_size_text=True,
                            auto_size_buttons=True,
                            resizable=True,
                            # default_element_size=(40, 1),
                            icon='../icons/kidney.ico',
                            finalize=True
                            )
    main_window.maximize()
    while True:
        event, values = main_window.read()

        if event in (buttons[0], 'Open'):
            global_messages.clear()
            data_path = gi.get_dicom_path(default_input_dicom_folder)  # get input dir
            if data_path != '':
                output_path = gi.get_images_path(default_output_folder)  # get output dir
                if output_path != '':
                    gi.main(data_path, output_path)
                else:
                    sg.popup_error(list_text[10] + "для сохранения снимков", title=titles[2],
                                   auto_close=True,
                                   auto_close_duration=3,
                                   background_color='brown')

            else:
                sg.popup_error(list_text[10] + list_text[8][6:], title=titles[2],
                               auto_close=True,
                               auto_close_duration=3,
                               background_color='brown')

        elif event in (buttons[1], 'View slice', 'Images for detect'):
            # start_folder = str(Path.cwd().parent / 'out')
            start_folder = default_output_folder
            input_path = gi.get_images_path(start_folder)
            if input_path != '' and os.path.exists(input_path):
                img_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(str(img_format))]
            if input_path == '' or img_files == []:
                warning(input_path, list_text[9])
            else:
                view_image_folder(input_path)

        elif event in (buttons[2], 'Detect stones'):
            start_folder = default_output_folder
            input_path = gi.get_images_path(start_folder)
            if input_path != '' and os.path.exists(input_path):
                detect_dir = input_path
                ds.detect_stones(detect_dir, save_confidence, path_to_yolo_weights)
                view_detected_images(detect_dir + '/detect/')

        elif event in (buttons[3], 'Detected images'):
            # start_folder = str(Path.cwd().parent / 'out')
            start_folder = default_output_folder
            input_path = gi.get_images_path(start_folder)
            if input_path != '' and os.path.exists(input_path):
                img_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(str(img_format))]
            if not img_files or input_path == '':
                warning(input_path, list_text[4])
            else:
                view_detected_images(input_path + '/detect/')

        elif event in (buttons[4], 'Detected images'):
            gs.main()

        elif event == buttons[14]:
            cf.main(lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder,
                    current_color_theme, img_count, img_format)
            main_window.close()
            sets()
            main()

        elif event in (sg.WIN_CLOSED, buttons[5]):  # close all end exit
            main_window.close()
            break

    return


if __name__ == "__main__":
    # suppress_qt_warnings()
    sets()
    main()
