import numpy as np
import matplotlib.pyplot as plt

# Parameters (example values)
alpha1 = 0.2
alpha2 = 0.2
beta1 = 0.03
beta2 = 0.2
dc = 0.01
db = 0.001
dfi = 0.008
xc = 0.001
xb = 0.001
gamma1 = 0.8
gamma2 = 0.3
gamma3 = 0.01
lamb = 1
g12 = -0.4
g21 = 0.1
k = 0.001  # time step
h = 0.001  # space step
T = 10 # final time
L = 1.0  # spatial domain

# Grid setup
nx = int(L / h) + 1
nt = int(T / k) + 1

# Initialize solution arrays
C = np.zeros((nx, nt))
B = np.zeros((nx, nt))
Fi = np.zeros((nx, nt))

start = int(0.2 / h)
end = int(0.8 / h)

#Variable values
C[start:end, 0] = 10
B[start:end, 0] = 50
Fi[start:end, 0] = 10

def stability():
    m = max(dc + ((gamma1 * xc) / beta1), db + (gamma2 * xb) / beta2)
    if m < dfi:
        print(m, dfi)
        print("stability failed")

# Update functions
def next_c(Fi, C, B, i, j):
    first_term = ((alpha1 * C[i][j] * (B[i][j] ** g12)) - (beta1 * C[i][j])) * k
    second_term = ((dc * k) / (h * h)) * (C[i-1][j] - 2 * C[i][j] + C[i+1][j])
    third_term = ((xc * Fi[i][j] * k) / (((lamb + Fi[i][j]) ** 2) * (h * h))) * \
                 (C[i+1][j] * (Fi[i+1][j] - Fi[i][j]) - C[i][j] * (Fi[i][j] - Fi[i-1][j]))
    fourth_term = ((xc * C[i][j] * k) / (h * h)) * \
                  ((Fi[i+1][j]) / ((lamb + Fi[i+1][j]) ** 2) - (Fi[i][j]) / ((lamb + Fi[i][j]) ** 2)) * (Fi[i+1][j] - Fi[i][j])
    return first_term + second_term + third_term + fourth_term + C[i][j]

def next_b(Fi, C, B, i, j):
    first_term = ((alpha2 * B[i][j] * (C[i][j] ** g21)) - (beta2 * B[i][j])) * k
    second_term = ((db * k) / (h * h)) * (B[i-1][j] - 2 * B[i][j] + B[i+1][j])
    third_term = ((xb * Fi[i][j] * k) / (((lamb + Fi[i][j]) ** 2) * (h * h))) * \
                 (B[i+1][j] * (Fi[i+1][j] - Fi[i][j]) - B[i][j] * (Fi[i][j] - Fi[i-1][j]))
    fourth_term = ((xb * B[i][j] * k) / (h * h)) * \
                  ((Fi[i+1][j]) / ((lamb + Fi[i+1][j]) ** 2) - (Fi[i][j]) / ((lamb + Fi[i][j]) ** 2)) * (Fi[i+1][j] - Fi[i][j])
    return first_term + second_term + third_term + fourth_term + B[i][j]

def next_fi(Fi, C, B, i, j):
    first_term = Fi[i][j] + k * (gamma1 * C[i][j] + gamma2 * B[i][j] - gamma3 * Fi[i][j])
    second_term = ((dfi * k) / (h * h)) * (Fi[i-1][j] - 2 * Fi[i][j] + Fi[i+1][j])
    return first_term + second_term

stability()

# Time-stepping
for j in range(nt - 1):
    for i in range(1, nx - 1):  # Skip boundaries

        if C[i, j] < 0 or B[i, j] < 0 or Fi[i, j] < 0:
            print(f"Invalid values detected at j={j}, i={i}: C={C[i, j]}, B={B[i, j]}, Fi={Fi[i, j]}")

        C[i][j+1] = next_c(Fi, C, B, i, j)
        B[i][j+1] = next_b(Fi, C, B, i, j)
        Fi[i][j+1] = next_fi(Fi, C, B, i, j)

# Function to plot heatmaps
def plot_heatmap(data, title, cmap="viridis"):
    plt.figure(figsize=(8, 4))
    plt.imshow(data.T, aspect='auto', origin='lower', cmap=cmap,
               extent=[0, L, 0, T])
    plt.colorbar(label='Concentration')
    plt.xlabel("Space (x)")
    plt.ylabel("Time (t)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot heatmaps
plot_heatmap(C, "Heatmap of C over space and time")
plot_heatmap(B, "Heatmap of B over space and time")
plot_heatmap(Fi, "Heatmap of Fi over space and time")

