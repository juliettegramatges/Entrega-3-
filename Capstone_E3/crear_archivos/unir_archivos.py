import pandas as pd

# Cargar los tres archivos CSV
df_plane_costs = pd.read_csv('../plane_costs.csv')
df_plane_dimensions = pd.read_csv('../plane_dimensions.csv')
df_flights = pd.read_csv('../flights.csv')


# Renombrar columnas de df_flights para que coincidan con las otras tablas
df_flights.rename(columns={'flight_type': 'Flight Type', 'plane_type': 'Plane Type'}, inplace=True)

# Verificar las columnas nuevamente
print("Columnas de plane_costs:", df_plane_costs.columns)
print("Columnas de plane_dimensions:", df_plane_dimensions.columns)
print("Columnas de flights:", df_flights.columns)

# Realizar el merge
df_flights_costs = pd.merge(df_flights, df_plane_costs, on=['Flight Type', 'Plane Type'], how='left')
df_flights_dimensions = pd.merge(df_flights_costs, df_plane_dimensions, on=['Flight Type', 'Plane Type'], how='left')

# Agregar una columna 'id' con secuencia numérica
df_flights_dimensions['id'] = range(1, len(df_flights_dimensions) + 1)

# Guardar el CSV final con la nueva columna 'id'
df_flights_dimensions.to_csv('../flights_combined.csv', index=False)
