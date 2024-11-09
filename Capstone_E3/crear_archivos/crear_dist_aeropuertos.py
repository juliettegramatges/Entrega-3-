import os
import csv

data = [
    ["EZE", "MIA", 7119],
    ["EZE", "BOG", 4689],
    ["EZE", "GRU", 1723],
    ["EZE", "LIM", 3153],
    ["EZE", "MEX", 7398],
    ["EZE", "SCL", 1139],
    ["MIA", "BOG", 2435],
    ["MIA", "GRU", 6574],
    ["MIA", "LIM", 4219],
    ["MIA", "MEX", 2051],
    ["MIA", "SCL", 6658],
    ["BOG", "GRU", 4336],
    ["BOG", "LIM", 1889],
    ["BOG", "MEX", 3159],
    ["BOG", "SCL", 4251],
    ["GRU", "LIM", 3475],
    ["GRU", "MEX", 7433],
    ["GRU", "SCL", 2614],
    ["LIM", "MEX", 4245],
    ["LIM", "SCL", 2462],
    ["MEX", "SCL", 6597]
]


# Obtener la ruta de la carpeta actual
current_folder = os.path.dirname(os.path.abspath(__file__))

# Obtener la ruta a la carpeta padre
parent_folder = os.path.dirname(current_folder)

# Nombre del archivo CSV
filename = os.path.join(parent_folder, "distances.csv")

# Escribir los datos en el archivo CSV
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    

    # Escribir las filas de datos
    writer.writerows(data)

print(f"Archivo CSV '{filename}' creado con éxito.")



# https://www.airportdistancecalculator.com/

