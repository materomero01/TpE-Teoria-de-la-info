import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calcular_estadisticas_columna_csv_sin_encabezado(rutas_archivos):
    resultados = {}
    for ruta_archivo in rutas_archivos:
        df = pd.read_csv(ruta_archivo, header=None)
        df.columns = ['Valores']
        df['Valores'] = pd.to_numeric(df['Valores'], errors='coerce')
        resultados[ruta_archivo] = {'media': df['Valores'].mean(), 'desvio_estandar': df['Valores'].std()}
    return resultados

def mostrar_resultados(resultados):
    for ruta, resultado in resultados.items():
        print(f"Resultados para {ruta}:\n   - Media: {resultado['media']:.2f}\n   - Desvío Estándar: {resultado['desvio_estandar']:.2f}\n")

def grafico_media(resultados):
    plt.figure(figsize=(10, 5))
    plt.bar(resultados.keys(), [resultado['media'] for resultado in resultados.values()], color='blue')
    plt.xlabel('Archivo')
    plt.ylabel('Media')
    plt.title('Media de los valores por Archivo')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def grafico_desvio_estandar(resultados):
    plt.figure(figsize=(10, 5))
    plt.bar(resultados.keys(), [resultado['desvio_estandar'] for resultado in resultados.values()], color='orange')
    plt.xlabel('Archivo')
    plt.ylabel('Desvío Estándar')
    plt.title('Desvío Estándar de los valores por Archivo')
    plt.xticks(rotation=45, ha='right')
    plt.show()

# Funciones para correlación cruzada
def calcular_correlacion_cruzada(senal1, senal2, muestras=50):
    """
    Calcula la correlación cruzada entre dos señales con muestreo aleatorio.

    Args:
        senal1 (pd.DataFrame): DataFrame con la señal 1.
        senal2 (pd.DataFrame): DataFrame con la señal 2.
        muestras (int): Número de muestras aleatorias para el cálculo (valor por defecto: 50).

    Returns:
        np.ndarray: Vector con la correlación cruzada entre las señales muestreadas.
    """
    muestra1 = senal1.sample(n=muestras, random_state=42)
    muestra2 = senal2.sample(n=muestras, random_state=42)
    valores1 = muestra1.values.squeeze()
    valores2 = muestra2.values.squeeze()
    return np.correlate(valores1, valores2, mode='full')

def calcular_correlacion_para_pares(pares_seniales, resultados_corr={}):
    """
    Calcula la correlación cruzada para cada par de señales con muestreo aleatorio.

    Args:
        pares_seniales (dict): Diccionario con pares de señales como claves (ej: ('S1.csv', 'S2.csv')) y None como valores.
        resultados_corr (dict): Diccionario para almacenar resultados (inicializar vacío).

    Returns:
        dict: Diccionario con los mismos pares de señales como claves y vectores de correlación cruzada como valores.
    """
    for par, _ in pares_seniales.items():
        senal1, senal2 = par
        resultados_corr[par] = calcular_correlacion_cruzada(senal1, senal2)
    return resultados_corr

def graficar_correlacion_por_par(resultados_corr):
    """
    Grafica la correlación cruzada para cada par de señales.

    Args:
        resultados_corr (dict): Diccionario con pares de señales como claves y vectores de correlación cruzada como valores.
    """
    for par, corr in resultados_corr.items():
        plt.figure(figsize=(10, 5))
        plt.plot(corr)
        plt.title(f"Correlación cruzada entre {par[0]} y {par[1]}")
        plt.xlabel("Desplazamiento")
        plt.ylabel("Correlación")
        plt.grid(True)
        plt.show()

# Definir archivos CSV
archivos = ['S1_buenosAires.csv', 'S2_bogota.csv', 'S3_vancouver.csv']

# Calcular estadísticas
resultados_estadisticas = calcular_estadisticas_columna_csv_sin_encabezado(archivos)

# Mostrar resultados
mostrar_resultados(resultados_estadisticas)

# Graficar media
grafico_media(resultados_estadisticas)

# Graficar desvío estándar
grafico_desvio_estandar(resultados_estadisticas)

# Definir pares de señales
pares_seniales = {('S1_buenosAires.csv', 'S2_bogota.csv'): None,
                  ('S1_buenosAires.csv', 'S3_vancouver.csv'): None,
                  ('S2_bogota.csv', 'S3_vancouver.csv'): None}

# Calcular correlación cruzada para pares
resultados_corr = calcular_correlacion_para_pares(pares_seniales)

# Graficar correlación cruzada para cada par
graficar_correlacion_por_par(resultados_corr)