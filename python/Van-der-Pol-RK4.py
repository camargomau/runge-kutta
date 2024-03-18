import numpy as np
import matplotlib.pyplot as plt

# Retorna la evaluación según la ecuación de Van der Pol
def van_der_pol(x, y, m):
    return [y, m*(1 - x**2)*y - x]

# Aplica RK4 para encontrar las soluciones a Van der Pol
def runge_kutta(t0, x0, y0, m, h, t_final):
    total_steps = int((t_final - t0) / h)

	# Arreglos para la discretización de t y los valores de x, y
    t = np.linspace(t0, t_final, total_steps)
    x = np.zeros(total_steps)
    y = np.zeros(total_steps)

    x[0] = x0
    y[0] = y0

	# Aproximación de los puntos hasta cubrir todos los valores de t
    for i in range(1, total_steps):
		# Pendientes
        k1 = h * np.array(van_der_pol(x[i-1], y[i-1], m))
        k2 = h * np.array(van_der_pol(x[i-1] + 0.5*k1[0], y[i-1] + 0.5*k1[1], m))
        k3 = h * np.array(van_der_pol(x[i-1] + 0.5*k2[0], y[i-1] + 0.5*k2[1], m))
        k4 = h * np.array(van_der_pol(x[i-1] + k3[0], y[i-1] + k3[1], m))

		# Aproximación del punto siguiente
        x[i] = x[i-1] + (1/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        y[i] = y[i-1] + (1/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

    return t, x, y

# Grafica la solución
def plot_solution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfica 1: x, y contra t
    ax1.plot(t, x, label="x(t)")
    ax1.plot(t, y, label="y(t)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x, y")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Solución para x, y de Van der Pol con RK4; independientes")

    # Gráfica 2: x contra y
    ax2.plot(x, y, label="Trayectoria")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Solución para x, y de Van der Pol con RK4; conjuntas")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Definición de μ, h y valores iniciales
# m = 1.0
# x0 = 1.0
# y0 = 0.0
# t0 = 0
# t_final = 50
# h = 0.01

m = float(input("• Introduce el valor del parámetro μ: "))
x0 = float(input("\n• Introduce el valor inicial x_0 para x: "))
y0 = float(input("• Introduce el valor inicial y_0 para y: "))
t0 = float(input("\n• Introduce el valor inical t_0 para t: "))
t_final = float(input("• Introduce el valor final que tomará t: "))
h = float(input("\n• Introduce el tamaño h de los pasos: "))

# Resolución con RK4 y graficación
t, x, y = runge_kutta(t0, x0, y0, m, h, t_final)
plot_solution()
