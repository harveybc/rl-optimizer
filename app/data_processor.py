import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import statsmodels.api as sm

# Directorio de salida
OUTPUT_DIR = 'output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Inicializamos la lista de resumen de resultados
summary_data = []

def analyze_dataset(dataset_path):
    try:
        # Cargar el dataset y limitar a las últimas 4500 filas
        df = pd.read_csv(dataset_path)
        df = df.tail(4500)
        
        # Extraer la cuarta columna (index 3) y convertir a número
        data_series = pd.to_numeric(df.iloc[:, 3], errors='coerce').dropna()

        # Cálculo de métricas
        mean_val = data_series.mean()
        std_val = data_series.std()
        snr_val = mean_val / std_val if std_val != 0 else 0
        normalized_error_std = np.sqrt(1 / snr_val) if snr_val != 0 else 0
        normalized_error_mean = (normalized_error_std * (np.sqrt(2 / np.pi))) if snr_val != 0 else 0
        
        # Autocorrelación para lags 1 a 10
        autocorrelation_vals = [data_series.autocorr(lag) for lag in range(1, 11)]

        # Análisis de Fourier
        N = len(data_series)
        T = 1.0  # Suponemos muestreo de 1 unidad de tiempo
        yf = fft(data_series - mean_val)  # Transformada de Fourier de la serie sin la media
        xf = fftfreq(N, T)[:N//2]
        power_spectrum = 2.0/N * np.abs(yf[:N//2])

        # Encontrar los 5 picos más potentes
        peaks, _ = find_peaks(power_spectrum, distance=20, prominence=0.1)
        top_peaks = sorted(zip(power_spectrum[peaks], xf[peaks]), reverse=True)[:5]

        # Guardar resultados en el resumen
        summary_data.append({
            'Dataset': os.path.basename(dataset_path),
            'Media': mean_val,
            'Desviación': std_val,
            'SNR': snr_val,
            'Desviación Error Normalizado': normalized_error_std,
            'Media Error Normalizado': normalized_error_mean,
            **{f'Autocorrelación lag {i+1}': autocorrelation_vals[i] for i in range(10)},
            **{f'Pico Fourier {i+1}': top_peaks[i][0] if i < len(top_peaks) else 'N/A' for i in range(5)},
            **{f'Frecuencia Pico {i+1}': top_peaks[i][1] if i < len(top_peaks) else 'N/A' for i in range(5)}
        })

        # Generar gráficos
        # Gráfico de estacionalidad, tendencia, y residuo
        decomposition = sm.tsa.seasonal_decompose(data_series, model='additive', period=30)
        decomposition.plot()
        plt.suptitle(f'Descomposición de la Serie - {os.path.basename(dataset_path)}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{os.path.basename(dataset_path)}_descomposicion.png'))
        plt.close()

        # Espectro de Fourier
        plt.plot(xf, power_spectrum)
        plt.title(f'Espectro de Fourier - {os.path.basename(dataset_path)}')
        plt.xlabel('Frecuencia')
        plt.ylabel('Amplitud')
        for _, peak_freq in top_peaks:
            plt.axvline(x=peak_freq, color='r', linestyle='--')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{os.path.basename(dataset_path)}_fourier_spectrum.png'))
        plt.close()

    except Exception as e:
        print(f'[ERROR] Ocurrió un error analizando {dataset_path}: {e}')

# Iterar sobre los datasets
datasets = [
    'datasets/eur-usd-historical-daily-data-test.csv',
    # Añadir aquí otros datasets si es necesario
]

for dataset in datasets:
    analyze_dataset(dataset)

# Guardar resumen en un archivo CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'summary_metrics.csv'), index=False)

# Mostrar el resumen
print(summary_df)

# Mostrar la tabla resumen final
print("\nResumen final de todas las métricas calculadas:\n")
print(summary_df.to_string(index=False))
