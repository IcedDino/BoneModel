import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# ====== NUMERICAL PARAMETERS ======
# Spatial step size
h = 0.1  # dx in original code
# Time step size
k = 0.1  # dt in original code
# Length of time
time = 900
# Adjusted time steps for time
time_steps = int(time / k)

# ====== PROBLEM PARAMETERS ======
alpha = [0.2, 0.2]  # alpha1, alpha2 (for C and B)
beta1 = 0.03  # beta1 (for C)
beta2 = 0.2  # beta2 (for B)
g12 = -0.4  # g12 (coupling exponent from B to C)
g21 = 0.1  # g21 (coupling exponent from C to B)
dc = 0.045  # Diffusion coefficient for C
db = 0.0001  # Diffusion coefficient for B
dfi = 0.008  # Diffusion coefficient for Fi
lamb = 1  # lambda (regularization parameter)
xc = 0.015  # Chemotaxis coefficient for C
xb = 0.0008  # Chemotaxis coefficient for B
gamma1 = 0.8  # Production rate of Fi by C
gamma2 = 0.3  # Production rate of Fi by B
gamma3 = 0.01  # Decay rate of Fi

# Store parameters in arrays for convenience
D = [dc, db, dfi]  # Diffusion coefficients
X = [xc, xb]  # Chemotaxis coefficients
gamma = [gamma1, gamma2, gamma3]  # Fi production/decay parameters
r = [alpha[0], alpha[1], gamma[3], X[0], X[1]]  # Rate parameters


# Discrete terms from the previous code
def rho(i):
    return math.sqrt(r[i] / D[i])


def fi(i):
    return (math.e ** r[i] * k - 1) / r[i]


def psi(i):
    if i == 2:
        return (4 / (math.sqrt(r[i] / D[i]) ** 2)) * math.sinh(math.sqrt(r[i] / D[i]) * (h / 2)) ** 2
    return (4 / rho(i) ** 2) * math.sin(rho(i) * (h / 2)) ** 2


def next_c(Fi, C, B, i, t):
    """Update function for C (first species)"""
    # Original version from request
    first_term = ((alpha[0] * C[i][t] * (B[i][t] ** g12)) - (beta1 * C[i][t])) * k
    second_term = ((dc * k) / (h * h)) * (C[i - 1][t] - 2 * C[i][t] + C[i + 1][t])
    third_term = ((xc * Fi[i][t] * k) / (((lamb + Fi[i][t]) ** 2) * (h * h))) * (
                C[i + 1][t] * (Fi[i + 1][t] - Fi[i][t]) - C[i][t] * (Fi[i][t] - Fi[i - 1][t]))
    fourth_term = ((xc * C[i][t] * k) / (h * h)) * (
                (Fi[i + 1][t]) / ((lamb + Fi[i + 1][t]) ** 2) - (Fi[i][t]) / ((lamb + Fi[i][t]) ** 2)) * (
                              Fi[i + 1][t] - Fi[i][t])
    return first_term + second_term + third_term + fourth_term + C[i][t]


def next_b(Fi, C, B, i, t):
    """Update function for B (second species)"""
    first_term = ((alpha[1] * B[i][t] * (C[i][t] ** g21)) - (beta2 * B[i][t])) * k
    second_term = ((db * k) / (h * h)) * (B[i - 1][t] - 2 * B[i][t] + B[i + 1][t])
    third_term = ((xb * Fi[i][t] * k) / (((lamb + Fi[i][t]) ** 2) * (h * h))) * (
                B[i + 1][t] * (Fi[i + 1][t] - Fi[i][t]) - B[i][t] * (Fi[i][t] - Fi[i - 1][t]))
    fourth_term = ((xb * B[i][t] * k) / (h * h)) * (
                (Fi[i + 1][t]) / ((lamb + Fi[i + 1][t]) ** 2) - (Fi[i][t]) / ((lamb + Fi[i][t]) ** 2)) * (
                              Fi[i + 1][t] - Fi[i][t])
    return first_term + second_term + third_term + fourth_term + B[i][t]


def next_fi(Fi, C, B, i, t):
    """Update function for Fi (signaling molecule)"""
    first_term = Fi[i][t] + k * (gamma1 * C[i][t] + gamma2 * B[i][t] - gamma3 * Fi[i][t])
    second_term = ((dfi * k) / (h * h)) * (Fi[i - 1][t] - 2 * Fi[i][t] + Fi[i + 1][t])
    return first_term + second_term


def run_simulation(grid_size=50, time_steps=500):
    """Run the simulation and return the results"""
    # Initialize the grids
    Fi = np.zeros((grid_size, grid_size))
    C = np.zeros((grid_size, grid_size))
    B = np.zeros((grid_size, grid_size))

    # Set initial conditions (small random perturbations)
    C[grid_size // 4:3 * grid_size // 4, grid_size // 4:3 * grid_size // 4] = 0.1 + 0.01 * np.random.rand(
        grid_size // 2, grid_size // 2)
    B[grid_size // 4:3 * grid_size // 4, grid_size // 4:3 * grid_size // 4] = 0.1 + 0.01 * np.random.rand(
        grid_size // 2, grid_size // 2)

    # Storage for results
    results = []
    results.append((Fi.copy(), C.copy(), B.copy()))

    # Main simulation loop
    for t in range(time_steps):
        Fi_new = np.zeros_like(Fi)
        C_new = np.zeros_like(C)
        B_new = np.zeros_like(B)

        # Update interior points
        for i in range(1, grid_size - 1):
            for j in range(1, grid_size - 1):
                Fi_new[i][j] = next_fi(Fi, C, B, i, j)
                C_new[i][j] = next_c(Fi, C, B, i, j)
                B_new[i][j] = next_b(Fi, C, B, i, j)

        # Apply boundary conditions (no-flux)
        Fi_new[0, :] = Fi_new[1, :]
        Fi_new[-1, :] = Fi_new[-2, :]
        Fi_new[:, 0] = Fi_new[:, 1]
        Fi_new[:, -1] = Fi_new[:, -2]

        C_new[0, :] = C_new[1, :]
        C_new[-1, :] = C_new[-2, :]
        C_new[:, 0] = C_new[:, 1]
        C_new[:, -1] = C_new[:, -2]

        B_new[0, :] = B_new[1, :]
        B_new[-1, :] = B_new[-2, :]
        B_new[:, 0] = B_new[:, 1]
        B_new[:, -1] = B_new[:, -2]

        # Update the grids
        Fi = Fi_new
        C = C_new
        B = B_new

        # Store results every 10 time steps
        if t % 10 == 0:
            results.append((Fi.copy(), C.copy(), B.copy()))
            print(f"Completed time step {t}/{time_steps}")

    return results


def visualize_results(C, B, Fi, x_values):
    """Visualize the simulation results using the format from the previous code"""
    # Create master figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.7], width_ratios=[1, 1, 1])

    # Prepare data for heatmaps
    graph_parameters = [0, 1, 0, time]

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
        ax.set_xlabel("Spatial Position (x)")
        ax.set_ylabel("Time (t)")
        heatmap_axes.append(ax)

    # Create time series subplot (spanning entire bottom row)
    ax_ts = fig.add_subplot(gs[1, :])

    # Generate time indices and flatten data
    time_indices = np.repeat(np.arange(time_steps), len(x_values))
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


def animate_results(results, x_values):
    """Create an animation of the simulation results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    Fi, C, B = results[0]

    # Create line plots instead of imshow for 1D data
    line1, = axes[0].plot(x_values, Fi, color='blue')
    axes[0].set_ylim(0, np.max([r[2][0] for r in results]) * 1.1)
    axes[0].set_title('Signaling Molecule (Fi)')

    line2, = axes[1].plot(x_values, C, color='red')
    axes[1].set_ylim(0, np.max([r[1][0] for r in results]) * 1.1)
    axes[1].set_title('Osteoclast Population (C)')

    line3, = axes[2].plot(x_values, B, color='green')
    axes[2].set_ylim(0, np.max([r[2][0] for r in results]) * 1.1)
    axes[2].set_title('Osteoblast Population (B)')

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    def update(frame):
        Fi, C, B = results[frame]
        line1.set_ydata(Fi)
        line2.set_ydata(C)
        line3.set_ydata(B)
        return [line1, line2, line3]

    ani = FuncAnimation(fig, update, frames=len(results), blit=True, interval=100)
    plt.show()
    return ani


# Run the simulation
if __name__ == "__main__":
    print("Starting simulation...")
    C, B, Fi, x_values, results = run_simulation()
    print("Simulation complete.")

    # Visualize the results
    visualize_results(C, B, Fi, x_values)

    # Create an animation of the spatial profiles over time
    ani = animate_results(results, x_values)
    # To save the animation (requires ffmpeg):
    # ani.save('simulation.mp4', writer='ffmpeg', fps=10)