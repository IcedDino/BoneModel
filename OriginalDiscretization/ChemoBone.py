import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha_1 = 0.2
alpha_2 = 0.2

beta_1 = 0.03
beta_2 = 0.2

g_12 = -0.4
g_21 = 0.1

Dc = 0.01
Db = 0.001
DFi = 0.008

h = 0.1  # Spatial step size
dt = 0.01  # Time step size

lamb = 2

Xc = 0.001
Xb = 0.001

gamma_1 = 0.8
gamma_2 = 0.3
gamma_3 = 0.01

time = 900
time_steps = int(time / dt)  # Number of time steps

# Spatial grid (0 to 1, step size h)
x_values = np.arange(0, 1 + h, h)
grid_size = len(x_values)

eval_start = int(0 / h)
eval_end = int(0.9999 / h) + 1

# Initialize arrays
C = np.zeros((grid_size, time_steps))
B = np.zeros((grid_size, time_steps))
Fi = np.zeros((grid_size, time_steps))

start = int(0.2 / h)
end = int(0.8 / h)

C[start:end, 0] = 10
B[start:end, 0] = 50
Fi[start:end, 0] = 10

stabilityLambda_a = Dc * (dt / h * h)
stabilityLambda_b = Db * (dt / h * h)
stabilityLambda_c = DFi * (dt / h * h)

print(f"Lambda Dc = {stabilityLambda_a}, Lambda Db = {stabilityLambda_b}, Lambda Dfi = {stabilityLambda_c}")

def chemotaxis(i, t, param):
    # Forward differences for chemotaxis terms

    d2Fi_dx2 = (Fi[i + 1, t] - 2 * Fi[i, t] + Fi[i - 1, t]) / (h * h) #Fi Second derivative

    FiC = ((Fi[i, t] * param[i, t]) / ((lamb + Fi[i, t]) ** 2)) #Fi_Lambda first derivative

    dFiC = ((Fi[i + 1, t] * param[i + 1, t]) / ((lamb + Fi[i + 1, t]) ** 2)) - ((Fi[i, t] * param [i, t]) / ((lamb + Fi[i, t]) ** 2))#Fi_C first derivative

    dFi_dx = (Fi[i + 1, t] - Fi[i, t]) / h #Fi first derivative

    # Chemotaxis terms
    if Fi[i, t] == 0 or param[i, t] == 0:
        fi_term_a = 0
        fi_term_b = 0
    else:
        fi_term_a = FiC * d2Fi_dx2
        fi_term_b = (1/h) * dFiC * dFi_dx

    return fi_term_a + fi_term_b


def update_c(i, t):
    if alpha_1 * C[i, t] * B[i, t] == 0:
        linear_terms = 0 - (beta_1 * C[i, t])
    else:
        linear_terms = (alpha_1 * C[i, t] * (B[i, t] ** g_12)) - (beta_1 * C[i, t])

    diffusion_term = (Dc / (h * h)) * (C[i - 1, t] - 2 * C[i, t] + C[i + 1, t])

    fi_terms = chemotaxis(i, t, B)

    chemotaxis_terms = Xc * fi_terms

    return C[i, t] + dt * (linear_terms + diffusion_term - chemotaxis_terms)


def update_b(i, t):
    if alpha_2 * C[i, t] * B[i, t] == 0:
        linear_terms = 0 - (beta_2 * B[i, t])
    else:
        linear_terms = (alpha_2 * (C[i, t] ** g_21) * B[i, t]) - (beta_2 * B[i, t])

    diffusion_term = (Db / (h * h)) * (B[i - 1, t] - 2 * B[i, t] + B[i + 1, t])

    fi_terms = chemotaxis(i, t, B)

    chemotaxis_terms = Xb * fi_terms

    return B[i, t] + dt * (linear_terms + diffusion_term - chemotaxis_terms)


def update_fi(i, t):

    #Second derivative of Fi
    d2Fi_dx = (DFi / (h * h)) * (Fi[i - 1, t] - 2 * Fi[i, t] + Fi[i + 1, t])

    #Fi[i+1, t]
    return Fi[i, t] + dt * ((gamma_1 * C[i, t]) + (gamma_2 * B[i, t]) - (gamma_3 * Fi[i, t]) + d2Fi_dx)


def main():
    for t in range(time_steps - 1):
        for i in range(1, eval_end):
            if C[i, t] < 0 or B[i, t] < 0 or Fi[i, t] < 0:
                print(f"Invalid values detected at t={t}, i={i}: C={C[i, t]}, B={B[i, t]}, Fi={Fi[i, t]}")
            C[i, t + 1] = update_c(i, t)
            B[i, t + 1] = update_b(i, t)
            Fi[i, t+1] = update_fi(i, t)

    np.savetxt("matrizC.csv", C, delimiter=",")
    np.savetxt("matrizB.csv", C, delimiter=",")

    # Print some statistics to verify the simulation
    print(f"Max C value: {np.max(C)}")
    print(f"Min C value: {np.min(C)}")
    print(f"Mean C value: {np.mean(C)}")

    print(f"Max B value: {np.max(B)}")
    print(f"Min B value: {np.min(B)}")
    print(f"Mean B value: {np.mean(B)}")

    print(f"Max Fi value: {np.max(Fi)}")
    print(f"Min Fi value: {np.min(Fi)}")
    print(f"Mean Fi value: {np.mean(Fi)}")

    graph_parameters = [0, 1, 0, time]

    x_axis_label = "Spatial Position (x)"
    y_axis_label = "Time (t)"

    # Plot heatmap for C (Osteoclast)
    plt.figure(figsize=(10, 6))
    plt.imshow(C.T, aspect='auto', extent=graph_parameters, origin='lower', cmap='viridis')
    plt.colorbar(label="Osteoclast Population (C)")
    plt.title("Evolution of Osteoclast Population C")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

    # Plot heatmap for B (Osteoblast)
    plt.figure(figsize=(10, 6))
    plt.imshow(B.T, aspect='auto', extent=graph_parameters, origin='lower', cmap='viridis')
    plt.colorbar(label="Osteoblast Population (B)")
    plt.title("Evolution of Osteoblast Population B")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

    # Plot heatmap for Fi (Chemoattractant)
    plt.figure(figsize=(10, 6))
    plt.imshow(Fi.T, aspect='auto', extent=graph_parameters, origin='lower', cmap='viridis')
    plt.colorbar(label="Chemoattractant Population (Fi)")
    plt.title("Evolution of Chemoattractant Population Fi")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.show()

    # Plot time series for a range of spatial positions
    plt.figure(figsize=(12, 5))
    C_eval = C[:, :]  # All spatial positions over time
    B_eval = B[:, :]  # All spatial positions over time

    # Flatten arrays for plotting
    time_indices = np.tile(np.arange(time_steps), grid_size)
    c_values = C_eval.flatten()
    b_values = B_eval.flatten()

    plt.scatter(time_indices, c_values, s=1, alpha=0.7, label='C (Osteoclast Population)', color='blue')
    plt.scatter(time_indices, b_values, s=1, alpha=0.7, label='B (Osteoblast Population)', color='red')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Dynamics Over Time')
    plt.show()




if __name__ == "__main__":
    main()
