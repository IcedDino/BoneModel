import numpy as np
import matplotlib.pyplot as plt
from sympy.polys.matrices.sdm import sdm_matmul

# Parameters
alpha_1 = 0.2
alpha_2 = 0.2

beta_1 = 0.03
beta_2 = 0.2

g_12 = -0.4
g_21 = 0.1

Dc = 0.01
Db = 0.001

h = 0.1  # Spatial step size
dt = 0.01 # Time step size

time = 900
time_steps = int(time/dt)  # Number of time steps

# Spatial grid (0 to 1, step size h)
x_values = np.arange(0, 1 + h, h)
grid_size = len(x_values)

eval_start = int(0 / h)
eval_end = int(0.9999 / h) + 1

# Initialize arrays
C = np.zeros((grid_size, time_steps))
B = np.zeros((grid_size, time_steps))

start = int(0.2/h)
end = int(0.8/h)

print("vals")
print(start,end)

C[start:end, 0] = 10
B[start:end, 0] = 50


stabilityLambda_a = Dc * (dt / h * h)
stabilityLambda_b= Db * (dt / h * h)

print(stabilityLambda_a, stabilityLambda_b)


def update_c(i,t):
    if alpha_1 * C[i,t] * B[i,t] == 0:
        linear_terms = 0 - (beta_1 * C[i, t])
    else:
        linear_terms = (alpha_1 * C[i, t] * (B[i, t] ** g_12)) - (beta_1 * C[i, t])

    diffusion_term = (Dc / (h * h)) * (C[i - 1, t] - 2 * C[i, t] + C[i + 1, t])

    # Update C using forward Euler in time
    return C[i, t] + dt * (linear_terms + diffusion_term)

def update_b(i,t):
    if alpha_2 * C[i,t] * B[i,t] == 0:
        linear_terms = 0 - (beta_2 * B[i, t])
    else:
        linear_terms = (alpha_2 * (C[i, t] ** g_21) * B[i, t]) - (beta_2 * B[i, t])

    diffusion_term = (Db / (h * h)) * (B[i - 1, t] - 2 * B[i, t] + B[i + 1, t])

    # Update B using forward Euler in time
    return B[i, t] + dt * (linear_terms + diffusion_term)

def update_fi(i, t):
    return


def main():

    for t in range(time_steps - 1):
        for i in range(1, eval_end):
            if C[i, t] < 0 or B[i, t] < 0:
                print(f"Invalid values detected at t={t}, i={i}: C={C[i, t]}, B={B[i, t]}")
            C[i, t + 1] = update_c(i,t)
            B[i, t + 1] = update_b(i,t)


    np.savetxt("matrizC.csv", C, delimiter=",")
    np.savetxt("matrizB.csv", C, delimiter=",")

    # Plot heatmap for C (Osteoclast)
    plt.figure(figsize=(10, 6))
    plt.imshow(C.T, aspect='auto', extent=[0, 1, 0, time], origin='lower', cmap='viridis')
    plt.colorbar(label="Osteoclast Population (C)")
    plt.title("Evolution of Osteoclast Population C")
    plt.xlabel("Spatial Position (x)")
    plt.ylabel("Time (t)")
    plt.show()

    # Plot heatmap for B (Osteoblast)
    plt.figure(figsize=(10, 6))
    plt.imshow(B.T, aspect='auto', extent=[0, 1, 0, time], origin='lower', cmap='viridis')
    plt.colorbar(label="Osteoblast Population (B)")
    plt.title("Evolution of Osteoblast Population B")
    plt.xlabel("Spatial Position (x)")
    plt.ylabel("Time (t)")
    plt.show()

    # Plot time series for a range of spatial positions
    plt.figure(figsize=(12, 5))
    C_eval = C[:, :]  # All spatial positions over time
    B_eval = B[:, :]  # All spatial positions over time

    # Flatten arrays for plotting
    time_indices = np.tile(np.arange(time_steps), grid_size)
    C_values = C_eval.flatten()
    B_values = B_eval.flatten()

    plt.scatter(time_indices, C_values, s=1, alpha=0.7, label='C (Osteoclast Population)', color='blue')
    plt.scatter(time_indices, B_values, s=1, alpha=0.7, label='B (Osteoblast Population)', color='red')

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('Population Dynamics Over Time')
    plt.show()

# Print some statistics to verify the simulation
print(f"Max C value: {np.max(C)}")
print(f"Min C value: {np.min(C)}")
print(f"Mean C value: {np.mean(C)}")

print(f"Max B value: {np.max(B)}")
print(f"Min B value: {np.min(B)}")
print(f"Mean B value: {np.mean(B)}")

if __name__ == "__main__":
    main()
