import math
import numpy as np
import matplotlib.pyplot as plt

# ====== NUMERICAL PARAMETERS ======
#Spatial step size
dx = 0.1
#Time step size
dt = 0.1
#Lenght of time
time = 900
#Adjusted time steps for time
time_steps = int(time / dt)

# ====== PROBLEM PARAMETERS ======
alpha = [0.2, 0.2] #alpha1, alpha 2
beta = [0.03, 0.2] #beta1, beta2
g = [-0.4, 0.1] #g12, g21
D = [0.001, 0.0001, 0.1] #Dc, Db, DFi
l = 0.001 #lambda
X = [0.00001, 0.00001] #c, b
gamma = [0.8,0.3,0.01] #gamma1, gamma2, gamma3
r = [alpha[0], alpha[1], gamma[2], X[0], X[1]]

#====== GRID SETUP ======
x_values = np.arange(0, 1 + dx, dx)
grid_size = len(x_values)
eval_start = int(0 / dx)
eval_end = int(0.9999 / dx) + 1

#====== INITIAL CONDITIONS ======
# Initialize arrays
C = np.zeros((grid_size, time_steps))
B = np.zeros((grid_size, time_steps))
Fi = np.zeros((grid_size, time_steps))
#Define starting conditions bounds
start = int(0.2 / dx)
end = int(0.8 / dx)
#Variable values
C[start:end, 0] = 10
B[start:end, 0] = 50
Fi[start:end, 0] = 10

#====== STABILITY CHECKS ======
stabilityLambda_a = D[0] * (dt / dx * dx)
stabilityLambda_b = D[1] * (dt / dx * dx)
stabilityLambda_c = D[2] * (dt / dx * dx)
print(f"Lambda Dc = {stabilityLambda_a}, Lambda Db = {stabilityLambda_b}, Lambda Dfi = {stabilityLambda_c} \n")


#====== DISCRETE TERMS ======
def rho(i):
    return math.sqrt(r[i]/D[i])

def fi(i):
    return (math.e ** r[i] * dt - 1) / r[i]

def psi(i):

    if i == 2:
        return (4 / (math.sqrt(r[i] / D[i]) ** 2)) * math.sinh(math.sqrt(r[i] / D[i]) * (dx / 2)) ** 2

    return (4 / rho(i) ** 2) * math.sin(rho(i) * (dx / 2)) ** 2


#====== UPDATE FUNCTIONS ======
# type=0 for C, type=1 for B
def update_c(x, t, var, types):

    first_term = ((D[types] * fi(types)) / psi(types)) * (var[x + 1, t] - 2 * var[x, t] + var[x - 1, t])
    second_term = (fi(types) / psi(2)) * ((X[types] * Fi[x, t] * var[x, t]) / ((l + Fi[x,t]) ** 2)) * (Fi[x+1, t] - 2 * Fi[x,t] + Fi[x-1, t])
    third_term = ((X[types] * fi(types)) / psi(types)) * (((Fi[x + 1, t] * var[x + 1, t]) / (l + Fi[x + 1, t]) ** 2) - (
            (Fi[x, t] * var[x, t]) / (l + Fi[x, t]) ** 2)) * ((Fi[x + 1, t] - Fi[x, t]) / psi(2))

    term_sum = first_term - second_term - third_term

    return term_sum / (1 - (alpha[types] * B[x, t] ** g[types] + beta[types])) * fi(types)


def update_b(x, t, var, types):

    first_term = ((D[types] * fi(types)) / psi(types)) * (var[x + 1, t] - 2 * var[x, t] + var[x - 1, t])
    second_term = (fi(types) / psi(2)) * (X[types] * Fi[x, t] * var[x, t] / ((l + Fi[x,t]) ** 2)) * (Fi[x + 1, t] - 2 * Fi[x,t] + Fi[x - 1, t])
    third_term = ((X[types] * fi(types)) / psi(types)) * (((Fi[x + 1, t] * var[x + 1, t]) / (l + Fi[x + 1, t]) ** 2) - (
            (Fi[x, t] * var[x, t]) / (l + Fi[x, t]) ** 2)) * ((Fi[x + 1, t] - Fi[x, t]) / psi(2))

    term_sum = first_term - second_term - third_term

    return term_sum / (1 - (alpha[types] * C[x, t] ** g[types] + beta[types]))


def update_fi(i, t):

    first_term = fi(2) * gamma[0] * C[i, t]
    second_term = fi(2) * gamma[1] * B[i, t]
    third_term = (D[2] * fi(2) / psi(2)) * (Fi[i+1, t] - 2 * Fi[i, t] + Fi[i-1, t])

    term_sum = first_term + second_term + third_term / (1+gamma[2])

    return term_sum

#====== GRAPHING ======
def graph(graph_parameters, x_axis_label, y_axis_label, time_steps, grid_size):
    # Create master figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.7], width_ratios=[1, 1, 1])

    # Common parameters for heatmaps
    heatmap_data = [
        (C.T, "Osteoclast Population (C)", "Evolution of C"),
        (B.T, "Osteoblast Population (B)", "Evolution of B"),
        (Fi.T, "Chemoattractant (Fi)", "Evolution of Fi")
    ]

    # Create heatmap subplots
    heatmap_axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        data, cb_label, title = heatmap_data[i]
        im = ax.imshow(data, aspect='auto', extent=graph_parameters,
                       origin='lower', cmap='viridis')
        plt.colorbar(im, ax=ax, label=cb_label, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        heatmap_axes.append(ax)

    # Create time series subplot (spanning entire bottom row)
    ax_ts = fig.add_subplot(gs[1, :])

    # Generate time indices and flatten data
    time_indices = np.repeat(np.arange(time_steps), grid_size)
    c_values = C.ravel()
    b_values = B.ravel()

    # Create scatter plots
    ax_ts.scatter(time_indices, c_values, s=1, alpha=0.7,
                  label='C (Osteoclast)', color='blue')
    ax_ts.scatter(time_indices, b_values, s=1, alpha=0.7,
                  label='B (Osteoblast)', color='red')

    ax_ts.legend(markerscale=5)
    ax_ts.set_xlabel('Time')
    ax_ts.set_ylabel('Population')
    ax_ts.set_title('Population Dynamics Over Time')
    ax_ts.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():

    #====== UPDATE ======
    for t in range(time_steps - 1):
        for i in range(1, eval_end):

            #Error check
            if C[i, t] < 0 or B[i, t] < 0 or Fi[i, t] < 0:
                print(f"Invalid values detected at t={t}, i={i}: C={C[i, t]}, B={B[i, t]}, Fi={Fi[i, t]}")

            C[i, t + 1] = update_c(i, t, C, 0)
            B[i, t + 1] = update_b(i, t, B, 1)
            Fi[i, t + 1] = update_fi(i, t)

    #Result saver
    np.savetxt("matrizC.csv", C, delimiter=",")
    np.savetxt("matrizB.csv", C, delimiter=",")

    #====== ERROR CHECK ======
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

    #====== GRAPHING ======
    graph_parameters = [0, 1, 0, time]
    x_axis_label = "Spatial Position (x)"
    y_axis_label = "Time (t)"

    graph(graph_parameters, x_axis_label, y_axis_label, time_steps, grid_size)

if __name__ == "__main__":
    main()
