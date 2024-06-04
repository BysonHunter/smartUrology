import yaml
import PySimpleGUI as sg
from pathlib import PurePath
# import kidney_main_GUI as km
from modules import interface

languages = ["Русский", "English"]
img_formats = ['png', 'jpg']

color_themes = ["Black", "BlueMono", "BluePurple", "BrightColors", "BrownBlue",
                "Dark", "Dark2", "DarkAmber", "DarkBlack", "DarkBlue",
                "DarkBrown", "DarkGreen", "DarkPurple", "DarkRed", "DarkTanBlue", "DarkTeal", "Default",
                "Green", "GreenMono", "GreenTan", "HotDogStand", "Kayak", "LightBlue", "LightBrown",
                "LightGreen", "LightGrey", "LightPurple", "LightTeal", "LightYellow", "Material1", "Material2",
                "NeutralBlue", "Purple", "Reddit", "Reds", "SandyBeach", "SystemDefault", "SystemDefault1",
                "SystemDefaultForReal", "Tan", "TanBlue", "TealMono", "Topanga"]


def read_yaml(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        return settings


def set_language(lang_of_interface):  # set the language of interface
    global tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles
    if lang_of_interface == 'Русский':
        tooltips = interface.RU_TOOLTIPS
        buttons = interface.RU_BUTTONS
        list_text = interface.RU_LIST_TEXT
        checkbox = interface.RU_CHECKBOX
        main_menu = interface.RU_MENU
        window_heads = interface.RU_WINDOW_HEADS
        titles = interface.RU_TITLES
    elif lang_of_interface == 'English':
        tooltips = interface.EN_TOOLTIPS
        buttons = interface.EN_BUTTONS
        list_text = interface.EN_LIST_TEXT
        checkbox = interface.EN_CHECKBOX
        main_menu = interface.EN_MENU
        window_heads = interface.EN_WINDOW_HEADS
        titles = interface.EN_TITLES
    return tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles


def read_settings():
    settings = read_yaml('./config/config.yaml')  # read settings from file

    lang_of_interface = str(settings[0].get('lang_of_interface'))
    path_to_yolo_weights = str(settings[1].get('path_to_yolo_weights'))
    default_input_dicom_folder = str(settings[2].get('default_input_dicom_folder'))
    default_output_folder = str(settings[3].get('default_output_folder'))
    current_color_theme = str(settings[4].get('current_color_theme'))
    img_count = str(settings[5].get('image_count'))
    img_format = str(settings[6].get('image_format'))

    return lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder, \
           current_color_theme, img_count, img_format


def save_settings(lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder,
                  current_color_theme, img_count, img_format):
    settings = [
        {"lang_of_interface": lang_of_interface},
        {"path_to_yolo_weights": path_to_yolo_weights},
        {"default_input_dicom_folder": default_input_dicom_folder},
        {"default_output_folder": default_output_folder},
        {"current_color_theme": current_color_theme},
        {"image_count": str(img_count)},
        {"image_format":str(img_format)}
    ]
    with open('./config/config.yaml', 'w', encoding="utf8") as file:
        yaml.dump(settings, file)
        file.close()


def main(lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder,
         current_color_theme, img_count, img_format):
    folder_model = str(PurePath(path_to_yolo_weights).parent)
    # tooltips, buttons, list_text, checkbox, main_menu, window_heads, titles = set_language(lang_of_interface)
    color = current_color_theme
    layout = [
        [sg.Text(list_text[26], size=(45, 1)),
         sg.Spin(values=languages,
                 size=(10, 1),
                 initial_value=lang_of_interface,
                 enable_events=True, readonly=True,
                 key='-LANG-')],
        [sg.Text(list_text[22], size=(45, 1)),
         sg.Input(key='-INPUT_FOLDER-'), sg.FolderBrowse(initial_folder=default_input_dicom_folder),
         ],
        [sg.Text(list_text[23], size=(45, 1)),
         sg.Input(key='-OUTPUT_FOLDER-'), sg.FolderBrowse(initial_folder=default_output_folder, ),
         ],
        [sg.Text(list_text[24], size=(45, 1)),
         sg.Input(key='-FILE_MODEL-'), sg.FilesBrowse(initial_folder=folder_model)
         ],
        [sg.Text(list_text[25], size=(45, 1)),
         sg.Spin(values=color_themes,
                 initial_value=color,
                 size=(20, 5),
                 readonly=True,
                 key='-COLOR-'),
         sg.Button(button_text=buttons[21], enable_events=True,
                   key='-def_scheme-', button_color='teal')],
        [sg.Text(text='Кол-во файлов для выгрузки', size=(45, 1)),
         sg.Input(default_text=img_count,
                  size=(20, 5),
                  key='-IMG_COUNT-'),
         ],
        [sg.Text(text='Формат изображений', size=(45, 1)),
         sg.Spin(values=img_formats,
                 initial_value=img_format,
                 size=(20, 5),
                 readonly=True,
                 enable_events=True,
                 key='-IMAGE_FORMAT-'),
         ],
        [sg.Button(button_text=buttons[20],
                   key='-SAVE_SETTINGS-',
                   enable_events=True),
         sg.Cancel(key='-DONT_SAVE_SETTINGS', button_color='teal')],
    ]

    settings_window = sg.Window("Настройка параметров системы", layout,
                                margins=(0, 0),
                                force_toplevel=True,
                                auto_size_text=True,
                                auto_size_buttons=True,
                                finalize=True
                                )

    settings_window['-LANG-'].update(lang_of_interface)
    settings_window['-FILE_MODEL-'].update(path_to_yolo_weights)
    settings_window['-INPUT_FOLDER-'].update(default_input_dicom_folder)
    settings_window['-OUTPUT_FOLDER-'].update(default_output_folder)
    settings_window['-COLOR-'].update(current_color_theme)
    settings_window['-IMG_COUNT-'].update(img_count)
    settings_window['-IMAGE_FORMAT-'].update(img_format)

    while True:
        event, values = settings_window.read()

        if event == '-def_scheme-':
            current_color_theme = 'SystemDefault'
            settings_window['-COLOR-'].update(current_color_theme)
            sg.theme(current_color_theme)

        # elif event == '-IMG_COUNT-':

        elif event == '-SAVE_SETTINGS-':
            lang_of_interface = values['-LANG-']
            path_to_yolo_weights = values['-FILE_MODEL-']
            default_input_dicom_folder = values['-INPUT_FOLDER-']
            default_output_folder = values['-OUTPUT_FOLDER-']
            current_color_theme = values['-COLOR-']
            img_count = values['-IMG_COUNT-']
            img_format = values['-IMAGE_FORMAT-']
            save_settings(lang_of_interface, path_to_yolo_weights, default_input_dicom_folder, default_output_folder,
                          current_color_theme, img_count, img_format)
            settings_window.close()
            return

        elif event in (sg.WIN_CLOSED, '-DONT_SAVE_SETTINGS'):  # close all end exit
            settings_window.close()
            return
