# Увязка координат скважин с сейсмической сеткой 

Задача реализована посредством следующих шагов:

- Загрузка данных формата XYZ (далее сейсмическая сетка)
- Сформировать таблицу данных матричного типа, где столбцами/индексами будут списки координат X/Y
- Загрузить скважины из переданного файла well_coord
- Используя Евклидово расстояние, для каждой скважины найти ближайший узел на сейсмиеской сетке.
- Построить тепловую карту и разместить на ней точки, определяющие положение скважин в пространстве.