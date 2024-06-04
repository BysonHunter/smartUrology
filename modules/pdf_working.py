import os.path
from fpdf import FPDF
from pathlib import Path


def create_PDF(stones_dir_path, RS_params, LS_params, param_numpy):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.add_font('DejaVu', '', r"./fonts/DejaVuSerifCondensed.ttf", uni=True)
    pdf.set_font('DejaVu', '', 8)
    width = 150
    height = 5
    col_width = pdf.w / 3.5
    row_height = pdf.font_size
    spacing = 1

    pdf.image(x=20, name=r"./icons/logo.png", w=150, h=20)
    pdf.cell(width, height, txt=f'', ln=1, align="C")
    research = Path(stones_dir_path).parts[-2]

    pdf.cell(width, height, txt=f'Пациент: {param_numpy[2]}, ID пациента {param_numpy[3]}', ln=1, align="L")
    pdf.cell(width, height, txt=f'Дата исследования КТ: {param_numpy[0]}', ln=1, align="L")
    pdf.cell(width, height, txt=f'Дата исследования поиска камней: {research}', ln=1, align="L")

    pdf.cell(width, height, txt=f'В правой почке найдено {len(RS_params)} камней', ln=1, align="L")
    if len(RS_params) > 0:
        pdf.cell(width, height, txt="Параметры камней", ln=1, align="L")
        for stone in range(len(RS_params)):
            r_data = [['размеры камня                    ', f'{RS_params[stone][2]:.2f} см Х {RS_params[stone][3]:.2f} '
                                                            f'см Х {RS_params[stone][4]:.2f} см'],
                      ['масса камня, грамм               ', f'{RS_params[stone][10]:.2f}'],
                      ['средняя плотность, гр/см3        ', f'{RS_params[stone][11]:.2f}'],
                      ['масса по средней плотности, грамм', f'{RS_params[stone][12]:.2f}'],
                      ['максимальная плотность по HU     ', f'{RS_params[stone][13]}'],
                      ['минимальная плотность по HU      ', f'{RS_params[stone][14]}'],
                      ['средняя плотность по HU          ', f'{RS_params[stone][15]:.0f}']
                      ]

            pdf.cell(width, height, txt=f'Правая почка. Камень № {stone + 1}', ln=1, align="L")
            pdf.image(stones_dir_path + '/stone_rk_' + str(stone) + '.png', w=50, h=50)
            x_pos = pdf.get_x()
            y_pos = pdf.get_y()
            if os.path.exists(stones_dir_path + '/stonerk_' + str(stone) + '_1.png'):
                pdf.image(x=x_pos + 50, y=y_pos - 50, name=stones_dir_path + '/stonerk_' + str(stone) + '_1.png', w=100,
                          h=50)
            for row in r_data:
                for item in row:
                    pdf.cell(col_width, row_height * spacing, txt=item, border=1)
                pdf.ln(row_height * spacing)
        pdf.ln(pdf.h)

    pdf.cell(width, height, txt=f'В левой почке найдено {len(LS_params)} камней', ln=1, align="L")
    if len(LS_params) > 0:
        pdf.cell(width, height, txt="Параметры камней", ln=1, align="L")
        for stone in range(len(LS_params)):
            l_data = [['размеры камня                    ', f'{LS_params[stone][2]:.2f} см Х {LS_params[stone][3]:.2f} '
                                                            f'см Х {LS_params[stone][4]:.2f} см'],
                      ['масса камня, грамм               ', f'{LS_params[stone][10]:.2f}'],
                      ['средняя плотность, гр/см3        ', f'{LS_params[stone][11]:.2f}'],
                      ['масса по средней плотности, грамм', f'{LS_params[stone][12]:.2f}'],
                      ['максимальная плотность по HU     ', f'{LS_params[stone][13]}'],
                      ['минимальная плотность по HU      ', f'{LS_params[stone][14]}'],
                      ['средняя плотность по HU          ', f'{LS_params[stone][15]:.0f}']
                      ]

            pdf.cell(width, height, txt=f'Левая почка. Камень № {stone + 1}', ln=1, align="L")
            pdf.image(stones_dir_path + '/stone_lk_' + str(stone) + '.png', w=50, h=50)
            x_pos = pdf.get_x()
            y_pos = pdf.get_y()
            if os.path.exists(stones_dir_path + '/stonelk_' + str(stone) + '_1.png'):
                pdf.image(x=x_pos + 50, y=y_pos - 50, name=stones_dir_path + '/stonelk_' + str(stone) + '_1.png', w=100,
                          h=50)
            for row in l_data:
                for item in row:
                    pdf.cell(col_width, row_height * spacing, txt=item, border=1)
                pdf.ln(row_height * spacing)
    pdfFileName = stones_dir_path + f'{param_numpy[3]}_{research}SI.pdf'
    if os.path.exists(pdfFileName):
        os.remove(pdfFileName)
    pdf.output(pdfFileName)
    return pdfFileName


def read_n_print_pdf(pdfFileName):
    import webbrowser
    file_to_open = r"file://" + str(pdfFileName)
    webbrowser.open_new(file_to_open)
