import numpy as np
import matplotlib.pyplot as plt

# Parámetro RRR, reemplaza esto con tus últimos 3 dígitos reales
RRR = 357  # <-- reemplaza por los tuyos

# 100 puntos de x entre (1 - 0.5 * RRR/1000) y (1 + 0.5 * RRR/1000)
delta = 0.5 * RRR / 1000
x = np.linspace(1 - delta, 1 + delta, 100)

# 18 valores de h logarítmicamente espaciados entre 1e-18 y 1e-1
h = np.logspace(-18, -1, 18)

# Creamos matrices 2D para evaluar cada combinación de x y h
X, H = np.meshgrid(x, h, indexing='ij')  # X.shape = (100, 18)

# Funciones
f_sin = np.sin(X)
f_log = np.log(X)

# --- DERIVADAS NUMÉRICAS ---

# Método de orden O(h)
deriv_Oh_sin = (np.sin(X + H) - np.sin(X - H)) / (2 * H)
deriv_Oh_log = (np.log(X + H) - np.log(X - H)) / (2 * H)

# Método de orden O(h^4)
deriv_Oh4_sin = (-np.sin(X + 2*H) + 8*np.sin(X + H) - 8*np.sin(X - H) + np.sin(X - 2*H)) / (12 * H)
deriv_Oh4_log = (-np.log(X + 2*H) + 8*np.log(X + H) - 8*np.log(X - H) + np.log(X - 2*H)) / (12 * H)

# --- DERIVADAS REALES ---
real_deriv_sin = np.cos(X)
real_deriv_log = 1 / X

# --- ERRORES ---
error_Oh_sin = np.abs(deriv_Oh_sin - real_deriv_sin)
error_Oh_log = np.abs(deriv_Oh_log - real_deriv_log)
error_Oh4_sin = np.abs(deriv_Oh4_sin - real_deriv_sin)
error_Oh4_log = np.abs(deriv_Oh4_log - real_deriv_log)

