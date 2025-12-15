import numpy as np
from scipy import integrate
from pymoc.utils import make_func, make_array, check_numpy_version


class ThermohalineColumn(object):
  r"""
  Vertical Advection-Diffusion Column Model with Temperature, Salinity and Buoyancy

  This class is a thermohaline extension of the original buoyancy-only ``Column`` model.
  It represents a 1D water column where temperature and salinity are prognostic tracers,
  and buoyancy is diagnosed via a linear equation of state:

  .. math::

      b = g\left[\alpha (T - T_0) - \beta (S - S_0)\right].

  The velocity profile is required as an input. The class can either compute the
  equilibrium profiles of temperature and salinity for a given vertical velocity
  profile and boundary conditions (via a boundary value problem), or compute
  tendencies and perform time-stepping for advection, diffusion, convection and
  horizontal advection.

  Boundary conditions:

  * Fixed temperature and salinity at the surface (``Ts``, ``Ss``) if no restoring is used.
  * Optional Stommel-style surface restoring for T and S with exchange coefficients
    ``lambda_T`` and ``lambda_S``; if these are not None, the surface values are no
    longer clamped but are relaxed toward ``Ts`` and ``Ss`` during timestepping.
  * Fixed T/S or fixed dT/dz, dS/dz at the bottom.

  Parameters
  ----------
  z : ndarray
      Vertical depth levels of column grid. Units: m
  kappa : float, function, or ndarray
      Vertical diffusivity profile. Units: m^2/s
  Ts : float, optional
      Surface temperature target / boundary value. Units: degC
  Tbot : float, optional
      Bottom temperature boundary condition (if ``Tzbot`` is None). Units: degC
  Tzbot : float, optional
      Bottom temperature gradient boundary condition (alternative to ``Tbot``).
      Units: degC/m
  Ss : float, optional
      Surface salinity target / boundary value. Units: psu
  Sbot : float, optional
      Bottom salinity boundary condition (if ``Szbot`` is None). Units: psu
  Szbot : float, optional
      Bottom salinity gradient boundary condition (alternative to ``Sbot``).
      Units: psu/m
  lambda_T : float, optional
      Stommel-style thermal exchange coefficient with the atmosphere. Units: 1/s.
      If None, the surface T is treated as a fixed Dirichlet boundary Ts.
  lambda_S : float, optional
      Stommel-style haline exchange coefficient with the atmosphere. Units: 1/s.
      If None, the surface S is treated as a fixed Dirichlet boundary Ss.
  T : float, function, or ndarray
      Initial vertical temperature profile. Units: degC
  S : float, function, or ndarray
      Initial vertical salinity profile. Units: psu
  Area : float, function, or ndarray
      Horizontal area of basin. Units: m^2
  N2min : float, optional
      Minimum stratification for convective adjustment in buoyancy units.
      Units: s^-2
  g : float, optional
      Gravitational acceleration. Units: m/s^2
  alpha : float, optional
      Thermal expansion coefficient. Units: 1/degC
  beta : float, optional
      Haline contraction coefficient. Units: 1/psu
  T0 : float, optional
      Reference temperature for EOS. Units: degC
  S0 : float, optional
      Reference salinity for EOS. Units: psu
  """

  def __init__(
      self,
      z=None,
      kappa=None,
      Ts=0.0,
      Tbot=0.0,
      Tzbot=None,
      Ss=35.0,
      Sbot=35.0,
      Szbot=None,
      lambda_T=None,
      lambda_S=None,
      T=0.0,
      S=35.0,
      Area=None,
      N2min=1e-7,
      g=9.81,
      alpha=2e-4,
      beta=8e-4,
      T0=0.0,
      S0=35.0
  ):
    # initialize grid
    if isinstance(z, np.ndarray) and len(z) > 0:
      self.z = z
    else:
      raise TypeError('z needs to be numpy array providing grid levels')

    self.kappa = make_func(kappa, self.z, 'kappa')
    self.Area = make_func(Area, self.z, 'Area')

    # EOS parameters
    self.g = g
    self.alpha = alpha
    self.beta = beta
    self.T0 = T0
    self.S0 = S0

    # Surface and bottom boundary conditions for T and S
    # Ts and Ss are both Dirichlet values (if no restoring) and atmospheric targets (if restoring)
    self.Ts = Ts
    self.Tbot = Tbot
    self.Tzbot = Tzbot

    self.Ss = Ss
    self.Sbot = Sbot
    self.Szbot = Szbot

    # Stommel-style restoring coefficients; if None, revert to hard Dirichlet at surface
    self.lambda_T = lambda_T
    self.lambda_S = lambda_S

    # Minimum stratification (in buoyancy units) for convective adjustment
    self.N2min = N2min

    # Prognostic tracers: T and S
    self.T = make_array(T, self.z, 'T')
    self.S = make_array(S, self.z, 'S')

    if check_numpy_version():
      self.Tz = np.gradient(self.T, self.z)
      self.Sz = np.gradient(self.S, self.z)
    else:
      self.Tz = 0. * self.z
      self.Sz = 0. * self.z

    # Diagnose buoyancy and its vertical gradient
    self.update_b()

  def update_b(self):
    r"""
    Update the buoyancy profile and its vertical gradient from T and S using
    the linear equation of state:

    .. math::

        b = g\left[\alpha (T - T_0) - \beta (S - S_0)\right].
    """
    self.b = self.g * (self.alpha * (self.T - self.T0) - self.beta * (self.S - self.S0))
    if check_numpy_version():
      self.bz = np.gradient(self.b, self.z)
    else:
      self.bz = 0. * self.z

    # define a surface buoyancy bs for use in convection (from Ts, Ss)
    self.bs = self.g * (self.alpha * (self.Ts - self.T0) - self.beta * (self.Ss - self.S0))

  def Akappa(self, z):
    r"""
    Compute the area integrated diffusivity :math:`A\kappa`
    at depth(s) z.
    """
    return self.Area(z) * self.kappa(z)

  def dAkappa_dz(self, z):
    r"""
    Compute the area integrated diffusivity gradient
    :math:`\partial_z\left(A\kappa\right)` at depth(s) z.
    """
    if not check_numpy_version():
      raise ImportError(
          'You need NumPy version 1.13.0 or later. Please upgrade your NumPy library.'
      )
    return np.gradient(self.Akappa(z), z)

  def bc(self, ya, yb):
    r"""
    Boundary conditions for the equilibrium advective-diffusive problem in T and S.

    State vector y = (T, dT/dz, S, dS/dz).

    At present, the equilibrium solver still uses Dirichlet surface conditions
    T(z=surface) = Ts and S(z=surface) = Ss, even if restoring is enabled
    in the time-stepping model.
    """
    # Temperature BC at bottom
    if self.Tzbot is None:
      bc_T_bottom = ya[0] - self.Tbot
    else:
      bc_T_bottom = ya[1] - self.Tzbot


    bc_T_surface = yb[0] - self.Ts


    # Salinity BC at bottom
    if self.Szbot is None:
      bc_S_bottom = ya[2] - self.Sbot
    else:
      bc_S_bottom = ya[3] - self.Szbot

    # Salinity BC at surface (Dirichlet for BVP)
    bc_S_surface = yb[2] - self.Ss

    return np.array([bc_T_bottom, bc_T_surface, bc_S_bottom, bc_S_surface])

  def ode(self, z, y):
    r"""
    ODE for equilibrium T and S profiles, to be solved as a boundary value problem.

    y = (T, dT/dz, S, dS/dz).
    """
    coeff = (self.wA(z) - self.dAkappa_dz(z)) / self.Akappa(z)

    dT_dz = y[1]
    d2T_dz2 = coeff * y[1]

    dS_dz = y[3]
    d2S_dz2 = coeff * y[3]

    return np.vstack((dT_dz, d2T_dz2, dS_dz, d2S_dz2))

  def solve_equi(self, wA):
    r"""
    Solve for the equilibrium T and S profiles given an area-integrated velocity
    profile wA and fixed surface/bottom boundary conditions for T and S.
    """
    self.wA = make_func(wA, self.z, 'w')
    sol_init = np.zeros((4, np.size(self.z)))

    sol_init[0, :] = self.T
    sol_init[2, :] = self.S

    if check_numpy_version():
      self.Tz = np.gradient(self.T, self.z)
      self.Sz = np.gradient(self.S, self.z)
    else:
      self.Tz = 0. * self.z
      self.Sz = 0. * self.z

    sol_init[1, :] = self.Tz
    sol_init[3, :] = self.Sz

    res = integrate.solve_bvp(self.ode, self.bc, self.z, sol_init)

    self.T = res.sol(self.z)[0, :]
    self.Tz = res.sol(self.z)[1, :]
    self.S = res.sol(self.z)[2, :]
    self.Sz = res.sol(self.z)[3, :]

    self.update_b()

  def vertadvdiff(self, wA, dt, do_conv=False):
    r"""
    Vertical advection and diffusion for T and S in the time-stepping solution.

    If lambda_T and lambda_S are None, the surface values are treated as hard
    Dirichlet (Ts, Ss) as in the original implementation. If they are non-None,
    the surface is allowed to evolve freely here and will be relaxed in timestep().
    """
    wA = make_array(wA, self.z, 'wA')
    dz = self.z[1:] - self.z[:-1]

    def _advdiff_tracer(q, q_surface, q_bottom, qz_bottom=None, enforce_surface_value=True):
      q = q.copy()

      # apply boundary conditions
      # upper Dirichlet only if no restoring is used and convection has not already imposed it
      if enforce_surface_value and not do_conv:
        q[-1] = q_surface

      # bottom BC: either fixed value or fixed gradient
      if qz_bottom is None:
        q[0] = q_bottom
      else:
        q[0] = q[1] - qz_bottom * dz[0]

      # vertical gradient
      qz = (q[1:] - q[:-1]) / dz
      qz_up = qz[1:]
      qz_down = qz[:-1]
      qzz = (qz_up - qz_down) / (0.5 * (dz[1:] + dz[:-1]))

      # upwind advection
      weff = wA - self.dAkappa_dz(self.z)
      qz_adv = qz_down.copy()
      qz_adv[weff[1:-1] < 0] = qz_up[weff[1:-1] < 0]

      dq_dt = (
          -weff[1:-1] * qz_adv / self.Area(self.z[1:-1]) +
          self.kappa(self.z[1:-1]) * qzz
      )
      q[1:-1] = q[1:-1] + dt * dq_dt

      return q

    enforce_T_surface = (self.lambda_T is None)
    enforce_S_surface = (self.lambda_S is None)

    self.T = _advdiff_tracer(
        self.T, self.Ts, self.Tbot, qz_bottom=self.Tzbot,
        enforce_surface_value=enforce_T_surface
    )

    self.S = _advdiff_tracer(
        self.S, self.Ss, self.Sbot, qz_bottom=self.Szbot,
        enforce_surface_value=enforce_S_surface
    )

    self.update_b()

  def convect(self):
    r"""
    Downward convective adjustment to remove static instability, mixing both
    temperature and salinity in the convective region.

    If lambda_T or lambda_S are None, the corresponding surface tracer is
    re-imposed to Ts or Ss at the surface. If restoring is used, the surface
    values are left free and will be nudged in timestep().
    """
    self.update_b()

    ind = self.b > self.bs

    if ind.any():
      zconv = np.max(self.z[np.invert(ind)]) if np.invert(ind).any() else self.z[0]

      dz = self.z[1:] - self.z[:-1]
      dz = np.append(dz, dz[-1])
      dz = np.insert(dz, 0, dz[0])
      dzc = 0.5 * (dz[1:] + dz[:-1])

      w = dzc * self.Area(self.z)

      w_ind = w[ind]
      w_tot = np.sum(w_ind)
      if w_tot > 0.0:
        T_mean = np.sum(self.T[ind] * w_ind) / w_tot
        S_mean = np.sum(self.S[ind] * w_ind) / w_tot

        self.T[ind] = T_mean
        self.S[ind] = S_mean

      if self.lambda_T is None:
        self.T[-1] = self.Ts
      if self.lambda_S is None:
        self.S[-1] = self.Ss

    else:
      if self.lambda_T is None:
        self.T[-1] = self.Ts
      if self.lambda_S is None:
        self.S[-1] = self.Ss

    self.update_b()

  def horadv(self, vdx_in, T_in, S_in, dt):
    r"""
    Horizontal advection of T and S into the column for the time-stepping solution.
    """
    vdx_in = make_array(vdx_in, self.z, 'vdx_in')
    T_in = make_array(T_in, self.z, 'T_in')
    S_in = make_array(S_in, self.z, 'S_in')

    adv_idx = vdx_in > 0.0

    dT = T_in - self.T
    self.T[adv_idx] = (
        self.T[adv_idx] +
        dt * vdx_in[adv_idx] * dT[adv_idx] / self.Area(self.z[adv_idx])
    )

    dS = S_in - self.S
    self.S[adv_idx] = (
        self.S[adv_idx] +
        dt * vdx_in[adv_idx] * dS[adv_idx] / self.Area(self.z[adv_idx])
    )

    self.update_b()

  def timestep(self, wA=0., dt=1., do_conv=False,
               vdx_in=None, T_in=None, S_in=None):
    r"""
    One timestep integration for T and S, including vertical advection, diffusion,
    convection, horizontal advection, and optional Stommel-style surface restoring.

    If lambda_T and lambda_S are non-None, the surface grid cell is relaxed toward
    Ts and Ss, respectively, with restoring tendencies:

      dT_surf/dt += lambda_T (Ts - T_surf)
      dS_surf/dt += lambda_S (Ss - S_surf)
    """
    if do_conv:
      self.convect()

    self.vertadvdiff(wA=wA, dt=dt, do_conv=do_conv)

    if vdx_in is not None:
      if (T_in is not None) and (S_in is not None):
        self.horadv(vdx_in=vdx_in, T_in=T_in, S_in=S_in, dt=dt)
      else:
        raise TypeError('T_in and S_in are needed if vdx_in is provided')

    # Stommel-ish surface restoring applied after advection and diffusion
    if self.lambda_T is not None and self.lambda_T > 0.0:
      self.T[-1] = self.T[-1] + dt * self.lambda_T * (self.Ts - self.T[-1])

    if self.lambda_S is not None and self.lambda_S > 0.0:
      self.S[-1] = self.S[-1] + dt * self.lambda_S * (self.Ss - self.S[-1])

    self.update_b()
