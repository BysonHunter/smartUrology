
    while table1.rowCount() > 0:
        table1.removeRow(0)

    while table1.columnCount() > 0:
        table1.removeColumn(0)


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

    for i in range(len(recommended)):
        table1.insertRow(i)
    for i in range(2):
        table1.insertColumn(i)
    for i1 in range(len(recommended)):
        for j1 in range(len(recommended[i1])):
            value = recommended[i1][j1]
            item1 = QTableWidgetItem(str(value))
            item1.setBackground(QBrush(QColor(*rec_colors[i1])))
            #num_of_color += 1
            # item.setForeground(QBrush(QColor(*color.text)))
            table1.setItem(i1, j1, item1)

