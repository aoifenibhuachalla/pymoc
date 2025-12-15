'''
This script is a variant of example_twocol_plusSO, demonstrating passive tracer injection after equilibration.
'''
from pymoc.modules import Psi_Thermwind, Psi_SO, Column
from pymoc.plotting import Interpolate_channel
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

# boundary conditions:
bs = 0.03
bs_north = 0.004
bmin = 0.0

# S.O. surface boundary conditions and grid:
l = 2.e6
y = np.asarray(np.linspace(0, l, 40))
tau = 0.13
bs_SO = (bs-bmin) * (y / y[-1])**2 + bmin

A_basin = 6e13
A_north = A_basin / 50.

# time-stepping parameters:
dt = 86400 * 30
MOC_up_iters = int(np.floor(2. * 360 * 86400 / dt))
plot_iters = int(np.ceil(300 * 360 * 86400 / dt))
total_iters = int(np.ceil(3000 * 360 * 86400 / dt))

kappa = 2e-5
z = np.asarray(np.linspace(-4000, 0, 80))

def b_basin(z):
    return bs * np.exp(z / 300.)
def b_north(z):
    return bs_north * np.exp(z / 300.)

AMOC = Psi_Thermwind(z=z, b1=b_basin, b2=b_north, f=1e-4)
AMOC.solve()
[Psi_iso_b, Psi_iso_n] = AMOC.Psibz()

SO = Psi_SO(
    z=z,
    y=y,
    b=b_basin(z),
    bs=bs_SO,
    tau=tau,
    f=1e-4,
    L=5e6,
    KGM=1000.,
    c=0.1,
    bvp_with_Ek=True
)
SO.solve()

basin = Column(z=z, kappa=kappa, Area=A_basin, b=b_basin, bs=bs, bbot=bmin)
north = Column(z=z, kappa=kappa, Area=A_north, b=b_north, bs=bs_north, bbot=bmin)

fig = plt.figure(figsize=(6, 10))
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
plt.ylim((-4e3, 0))
ax1.set_xlim((-10, 15))
ax2.set_xlim((-0.02, 0.03))
ax1.set_xlabel('$\\Psi$', fontsize=14)
ax2.set_xlabel('b', fontsize=14)

for ii in range(0, total_iters):
    wAb = (Psi_iso_b - SO.Psi) * 1e6
    wAN = -Psi_iso_n * 1e6
    basin.timestep(wA=wAb, dt=dt)
    north.timestep(wA=wAN, dt=dt, do_conv=True)
    if ii % MOC_up_iters == 0:
        AMOC.update(b1=basin.b, b2=north.b)
        AMOC.solve()
        [Psi_iso_b, Psi_iso_n] = AMOC.Psibz()
        SO.update(b=basin.b)
        SO.solve()
    if ii % plot_iters == 0:
        ax1.plot(AMOC.Psi, AMOC.z, linewidth=0.5, color='r')
        ax1.plot(SO.Psi, SO.z, linewidth=0.5, color='m')
        ax2.plot(basin.b, basin.z, linewidth=0.5, color='b')
        ax2.plot(north.b, north.z, linewidth=0.5, color='c')
        plt.pause(0.01)

# Inject tracer at surface in northern hemisphere after equilibration
y_surface = np.linspace(0, l, 40)
tracer_surface = (y_surface - y_surface[0]) / (y_surface[-1] - y_surface[0])
# Only inject in northern column at surface (z=0)
north.inject_tracer(surface_distribution=1.0)  # Set surface tracer to 1 at northernmost point
basin.inject_tracer(surface_distribution=0.0)  # Set surface tracer to 0 at equator

# Optionally, run a few more timesteps to see tracer evolution
for ii in range(100):
    wAb = (Psi_iso_b - SO.Psi) * 1e6
    wAN = -Psi_iso_n * 1e6
    basin.advect_diffuse_tracer(w=wAb, dt=dt)
    north.advect_diffuse_tracer(w=wAN, dt=dt)

# Extend the simulation to run for 100 years (assuming 360 days per year)
additional_iters = int(np.ceil(100 * 360 * 86400 / dt))

for ii in range(additional_iters):
    wAb = (Psi_iso_b - SO.Psi) * 1e6
    wAN = -Psi_iso_n * 1e6
    basin.advect_diffuse_tracer(w=wAb, dt=dt)
    north.advect_diffuse_tracer(w=wAN, dt=dt)

# Plot the final state of the model and tracer distribution
fig, axs = plt.subplots(1, 2, figsize=(12, 10))

# Plot overturning streamfunctions
ax1 = axs[0]
ax1.plot(AMOC.Psi, AMOC.z, label='AMOC', color='r')
ax1.plot(SO.Psi, SO.z, label='SO', color='m')
ax1.set_xlabel('$\\Psi$ [Sv]', fontsize=14)
ax1.set_ylabel('Depth [m]', fontsize=14)
ax1.set_title('Overturning Streamfunctions')
ax1.legend()

# Plot tracer distribution as a colormap
ax2 = axs[1]
tracer_distribution = np.vstack([basin.get_tracer_profile(), north.get_tracer_profile()])
img = ax2.imshow(tracer_distribution, aspect='auto', extent=[0, 1, -4000, 0], cmap='viridis')
ax2.set_xlabel('Column (0=Basin, 1=North)', fontsize=14)
ax2.set_ylabel('Depth [m]', fontsize=14)
ax2.set_title('Tracer Distribution')
fig.colorbar(img, ax=ax2, label='Tracer Concentration')

plt.tight_layout()
plt.show()
