# -------------------------------------------------------------------------------------------------------------
EN_TOOLTIPS = [' Read DICOM files and save images to output folder ...',  # 0
               ' View not detected images and Detect kidney`s and stones in images...',  # 1
               ' Detect kidney`s and stones in images ...',  # 2
               ' View ONLY detected images ...',  # 3
               ' Select detected images to store into traindataset ...',  # 4
               ' CLOSE PROGRAM AND EXIT ...',  # 5
               ' Check "V" to select all files into train dataset ',  # 6
               ' Delete current image and label file ...',  # 7
               ' Edit labels',  # 8
               ' Systemsettings',  # 9
               ]
RU_TOOLTIPS = [' Чтение данных КТ и формирование изображений ',  # 0
               ' Просмотр недетектированных изображений и Поиск камней в почках',  # 1
               ' Поиск камней в почках ',  # 2
               ' Просмотр детектированных изображений',  # 3
               ' Формирование обучающего датасета из детектитрованных изображений',  # 4
               ' Выход из программы ...',  # 5
               ' Нажмите, чтобы выбрать все изображения ... ',  # 6
               ' Удалить текущее изображение и файл с метками ...',  # 7
               'Просмотр и редактирование размеченных объектов',  # 8
               ' Настройки системы',  # 9
               ]
EN_BUTTONS = ['Read DICOM folder',  # 0
              'View images and Detect objects',  # 1
              'Detect objects',  # 2
              'View detected',  # 3
              'Form dataset',  # 4
              'Exit',  # 5
              '<<UP',  # 6
              '<Prv',  # 7
              'Nxt>',  # 8
              'DN>>',  # 9
              'Detect stones',  # 10
              'Delete',  # 11
              'Copy image to train dataset',  # 12
              'Browse',  # 13
              'SETTINGS',  # 14
              'Edit labels',  # 15
              'OK',  # 16
              'Stable model',  # 17
              'Experimental model',  # 18
              'Stones param',  # 19
              'Save and exit',  # 20
              'System scheme',  # 21
              ]
RU_BUTTONS = ['Чтение данных КТ',  # 0
              'Просмотр и поиск камней в почках',  # 1
              'Поиск камней в почках',  # 2
              'Просмотр детектированных изображений',  # 3
              'Формирование обучающего датасета',  # 4
              'ВЫХОД ',  # 5
              '<<Нч',  # 6
              '<Прд',  # 7
              'Слд>',  # 8
              'Кц>>',  # 9
              'Поиск камней',  # 10
              'Удалить',  # 11
              'Скопировать изображения в датасет',  # 12
              'Отрыть папку',  # 13
              'НАСТРОЙКИ',  # 14
              'Редактирование меток',  # 15
              'OK',  # 16
              'Стабильная модель',  # 17
              'Экспериментальная модель',  # 18
              'Параметры камней',  # 19
              'Сохранить и выйти',  # 20
              'Системная гамма',  # 21
              ]
EN_CHECKBOX = ['Select all images to train dataset',  # 0
               'Save confidence in label file',  # 1
               ]
RU_CHECKBOX = ['Выбрать все изображения в датасет',  # 0
               'Сохранять уверенность поиска в файле с метками',  # 1
               ]
EN_LIST_TEXT = ['Image Folder...',  # 0
                'Select a file. Use scroll-wheel or arrow keys on keyboard to scroll through files one by one.',  # 1
                'Choose an image from list on left:',  # 2
                'Error read label file... :( ',  # 3
                ' folder does not contain detected images and labels files!',  # 4
                'Label file...',  # 5
                'Select detected Image Folder...',  # 6
                'Please, select folder of images ...',  # 7
                'Source folder for DICOM files ',  # 8
                ' folder does not contain images',  # 9
                'Cancelled - No valid folder entered',  # 10
                'Reading DICOM dataset from directory',  # 11
                'Progress.....',  # 12
                'Writing images to',  # 13
                'Created folder',  # 14
                'Detecting stones in images from directory',  # 15
                'Progress..... detected current file is',  # 16
                'detected object save into',  # 17
                'Stone',  # 18
                'Right kidney',  # 19
                'Left kidney',  # 20
                'Slice',  # 21
                'Path to input DICOM folder',  # 22
                'Path to output folder',  # 23
                'Path to detect model',  # 24
                'Color scheme',  # 25
                'System language'  # 26
                ]
RU_LIST_TEXT = ['Папка с изображениями',  # 0
                'Для перебора файлов можно использовать клавиши со стрелками, колесо "мышки"',  # 1
                'Выберите файл из списка слева:',  # 2
                'Ошибка чтения файла с метками :( ...',  # 3
                ' папка не содержит детектированные снимки и файлы с метками! ',  # 4
                'Содержимое файла с метками ...',  # 5
                'Выберите папку с детектированными изображениями',  # 6
                'Выберите папку с изображениями',  # 7
                'Папка со снимками КТ',  # 8
                'папка не содержит изображений',  # 9
                'Прервано - не выбрана папка ',  # 10
                'Чтение данных КТ из выбранной папки: ',  # 11
                'Выполнение ....',  # 12
                'Сохраняем изображения в папку: ',  # 13
                'Создан каталог',  # 14
                'Поиск камней в почках на снимках из каталога ',  # 15
                'Выполнение... текущее изображение ',  # 16
                'найденные объекты сохранены в ',  # 17
                'Камень',  # 18
                'Правая почка',  # 19
                'Левая почка',  # 20
                'Срез',  # 21
                'Путь к папке с исходными DICOM файлами',  # 22
                'Путь к папке для сохранения изображений',  # 23
                'Путь к файлу с весами модели',  # 24
                'Цветовая схема',  # 25
                'Язык системы'  # 26
                ]
EN_MENU = [['Dicom Files', ['Open', 'View slice', 'Exit']],
           ['View', ['Images for detect', 'Detected images'], ],
           ['Detect', ['Detect stones']],
           ['Stones', ['Calc param']]
           ]

RU_MENU = [['Снимки КТ', ['Открыть', 'Просмотр', 'Выход']],
           ['Просмотр', ['Изображения для детектирования', 'Детектированные изображения'], ],
           ['Поиск камней', ['Детектирование снимков']],
           ['Камни', ['Расчет параметров']],
           ]
EN_WINDOW_HEADS = ['Get images from DICOM dataset',  # 0
                   'WARNING!!!!',  # 1
                   "Image Viewer",  # 2
                   'Info...',  # 3
                   'Stones info',  # 4
                   'Calc parameters of stone',  # 5
                   ]
RU_WINDOW_HEADS = ['Просмотр данных КТ и поиск камней в почках',  # 0
                   'ВНИМАНИЕ!!!',  # 1
                   'Просмотр изображений',  # 2
                   'Информация ...',  # 3
                   'Информация о камнях',  # 4
                   'Расчет параметров камней',  # 5
                   ]
EN_TITLES = ['DICOM folder',  # 0
             'Images folder',  # 1
             'Error',  # 2
             ]

RU_TITLES = ['Папка с КТ',  # 0
             'Папка с изображениями',  # 1
             'ОШИБКА',  # 2
             ]
