# coding: utf-8

import numpy as np
from scipy import integrate, optimize
from pymoc.utils import make_func


class TH_Psi_SO(object):
  r"""
  Southern Ocean Overturning Transport Model (Thermohaline-aware)

  Instances of this class represent a 1D model of the overturning transport in
  the Southern Ocean interior, calculated based on a buoyancy profile in the
  adjoining basin, and local surface buoyancy and surface wind stress in the SO.

  In a thermohaline setup, the basin buoyancy profile will typically be
  diagnosed from temperature and salinity in a :class:`ThermohalineColumn`
  via a linear equation of state, and passed in either directly (`b`) or via
  a column object (`col_basin`).

  Parameters
  ----------
  z : ndarray
      Vertical depth levels of overturning grid. Units: m
  y : ndarray
      Meridional overturning grid. Units: m
  col_basin : object, optional
      Column-like object (e.g. :class:`ThermohalineColumn`) that provides the
      buoyancy profile in the adjoining basin on the north side of the ACC.
      Must define attributes ``z`` and ``b``.
  b : float, function, or ndarray, optional
      Vertical buoyancy profile from the adjoining basin, on the north
      side of the ACC. Units: m/s^2
  bs : float, function, or ndarray, optional
      Surface level buoyancy boundary condition. Can be a constant,
      or an array or function in y. Units: m/s^2
  tau : float, function, or ndarray, optional
      Surface wind stress. Can be a constant, or an array or function
      in y. Units: N/m^2
  f : float, optional
      Coriolis parameter. Units s^-1
  rho : float, optional
      Density of sea water for Boussinesq approximation. Units: kg/m^3
  L : float, optional
      Zonal length of the modeled ACC. Units: m
  KGM : float, optional
      Gent & McWilliams (GM) eddy diffusivity coefficient. Units: m^2/s
  c : float, optional
      Phase speed cutoff for smoothing when solving the GM boundary value
      problem. Units: m/s
  bvp_with_Ek : bool, optional
      Whether to enforce the boundary condition that Psi_GM = -Psi_Ek at the
      ocean surface and bottom when solving the boundary value problem for
      the GM streamfunction.
  Hsill : float, optional
      Height (in m above ocean floor) of the "sill", where Psi_Ek is tapered.
  HEk : float, optional
      Depth of the surface Ekman layer. Units: m
  Htapertop : float, optional
      Depth of the quadratic surface tapering layer for the GM streamfunction.
      Units: m
  Htaperbot : float, optional
      Height of the quadratic bottom tapering layer for the GM streamfunction.
      Units: m
  smax : float, optional
      Maximum slope of the GM streamfunction, above which Psi_GM is clipped.
      Units: m^-1
  """

  def __init__(
      self,
      z=None,             # vertical grid (array, in)
      y=None,             # horizontal grid (array, in)
      col_basin=None,     # column object providing basin buoyancy (optional)
      b=None,             # buoyancy profile at northern end of ACC
      bs=None,            # surface buoyancy (function, array or float, in)
      tau=None,           # surface wind stress (function, array or float, in)
      f=1.2e-4,           # Coriolis parameter (in)
      rho=1030,           # Density of sea water (in)
      L=1e7,              # Zonal length of the ACC (in)
      KGM=1e3,            # GM coefficient (in)
      c=None,             # phase speed for F2010 BVP smoother of GM streamfunction
      bvp_with_Ek=False,  # if true, apply BC that Psi_GM=-Psi_Ek in F2010 BVP
      Hsill=None,         # height (in m above ocean floor) where Psi_Ek is tapered
      HEk=None,           # depth of surface Ekman layer
      Htapertop=None,     # quadratic tapering of GM at surface
      Htaperbot=None,     # quadratic tapering of GM at bottom
      smax=0.01,          # max slope for clipping of GM streamfunction
  ):

    # initialize vertical grid:
    if isinstance(z, np.ndarray):
      self.z = z
    else:
      raise TypeError('z needs to be numpy array providing grid levels')

    # initialize meridional grid:
    if isinstance(y, np.ndarray):
      self.y = y
    else:
      raise TypeError(
          'y needs to be numpy array providing horizontal grid (or boundaries) of ACC'
      )

    # store column reference (if provided)
    self.col_basin = col_basin

    # basin buoyancy profile b(z): can come from explicit b or from col_basin.b
    if b is not None:
      self.b = make_func(b, self.z, 'b')
    elif (self.col_basin is not None) and hasattr(self.col_basin, 'b'):
      # if column grid differs from SO grid, interpolate
      zb = np.asarray(self.col_basin.z)
      bb = np.asarray(self.col_basin.b)

      if np.allclose(zb, self.z):
        b_arg = bb
      else:
        def b_arg(zz, zb_=zb, bb_=bb):
          return np.interp(zz, zb_, bb_)
      self.b = make_func(b_arg, self.z, 'b')
    else:
      # default: zero buoyancy profile
      self.b = make_func(0.0, self.z, 'b')

    # surface buoyancy and wind stress
    self.bs = make_func(bs, self.y, 'bs')
    self.tau = make_func(tau, self.y, 'tau')

    # parameters
    self.f = f
    self.rho = rho
    self.L = L
    self.KGM = KGM
    self.c = c
    self.bvp_with_Ek = bvp_with_Ek
    self.Hsill = Hsill
    self.HEk = HEk
    self.Htapertop = Htapertop
    self.Htaperbot = Htaperbot
    self.smax = smax

    # placeholders for overturning components
    self.Psi_Ek = None
    self.Psi_GM = None
    self.Psi = None

  # --------------------------------------------------------------------------
  #  Inversion bs(y) -> y_s(b)
  # --------------------------------------------------------------------------
  def ys(self, b):
    r"""
    Inversion function of :math:`bs(y)`. This gives the outcropping latitude of
    the isopycnal of buoyancy class :math:`b`.

    Parameters
    ----------
    b : float
        The surface buoyancy value for whose meridional location is being
        calculated.

    Returns
    -------
    ys : float
        The meridional location at which the surface buoyancy bs equals b.
    """

    def func(y):
      return self.bs(y) - b

    if b < np.min(self.bs(self.y)):
      # if b is smaller than minimum bs, isopycnals don't outcrop and get handled separately
      return self.y[0] - 1e3
    if b > self.bs(self.y[-1]):
      # if b is larger than bs at northern end, return northernmost point:
      return self.y[-1]
    else:
      # if b in range of bs return ys(b):
      # Notice that this inversion is well defined only if bs is monotonically
      # increasing (past minind).
      minind = np.argmin(self.bs(self.y))
      return optimize.brentq(func, self.y[minind], self.y[-1])

  # --------------------------------------------------------------------------
  #  Buoyancy frequency N^2
  # --------------------------------------------------------------------------
  def calc_N2(self):
    r"""
    Calculate the buoyancy (Brunt-Väisälä) frequency profile for the Southern Ocean

    Returns
    -------
    N2 : function
         A depth-dependent function that returns the buoyancy frequency
         :math:`N^2` for a given depth :math:`z`.
    """

    dz = self.z[1:] - self.z[:-1]
    N2 = np.zeros(np.size(self.z))
    b = self.b(self.z)

    N2[1:-1] = (b[2:] - b[:-2]) / (dz[1:] + dz[:-1])
    N2[0] = (b[1] - b[0]) / dz[0]
    N2[-1] = (b[-1] - b[-2]) / dz[-1]

    return make_func(N2, self.z, 'N2')

  # --------------------------------------------------------------------------
  #  Bottom tapering
  # --------------------------------------------------------------------------
  def calc_bottom_taper(self, H, z):
    r"""
    Calculate the quadratic tapering profile relative to the ocean floor.

    Parameters
    ----------
    H : float
        Height above the bottom at which the streamfunction is tapered. Units: m
    z : ndarray
        Vertical depth levels of overturning grid. Units: m

    Returns
    -------
    bottom_taper : ndarray
                   Weights (0.0–1.0) corresponding to how much of the
                   streamfunction should remain after tapering.
    """

    if H is not None:
      return 1. - np.maximum(z[0] + H - z, 0.)**2. / H**2.
    return 1.

  # --------------------------------------------------------------------------
  #  Top tapering
  # --------------------------------------------------------------------------
  def calc_top_taper(self, H, z, scalar=True):
    r"""
    Calculate the quadratic tapering profile relative to the ocean surface.

    Parameters
    ----------
    H : float
        Depth from the surface at which the streamfunction is tapered. Units: m
    z : ndarray
        Vertical depth levels of overturning grid. Units: m
    scalar : bool, optional
        If True and H is None, return a scalar 1. If False and H is None, return
        an array of ones with zero at the surface grid point.

    Returns
    -------
    top_taper : ndarray or float
                Weights (0.0–1.0) corresponding to how much of the
                streamfunction should remain after tapering.
    """
    if H is not None:
      return 1 - np.maximum(z + H, 0)**2. / H**2.
    elif scalar:
      return 1.
    else:
      taper = np.ones(np.size(z))
      taper[-1] = 0.
      return taper

  # --------------------------------------------------------------------------
  #  Ekman transport
  # --------------------------------------------------------------------------
  def calc_Ekman(self):
    r"""
    Compute the Ekman transport from the wind stress averaged from the
    northern boundary of the domain to the latitude of the northernmost
    outcropped isopycnal.

    .. math::
      \Psi_{Ek} = -\frac{\tau L_x}{\rho_0 f_{SO}}

    Returns
    -------
    Psi_Ek : ndarray
             Meridional average of the Ekman transport at each vertical level
             of the Southern Ocean model (in m^3/s).
    """

    tau_ave = 0 * self.z
    for ii in range(0, np.size(self.z)):
      y0 = self.ys(self.b(self.z[ii]))  # outcrop latitude
      tau_ave[ii] = np.mean(self.tau(np.linspace(y0, self.y[-1], 100)))

    silltaper = self.calc_bottom_taper(self.Hsill, self.z)
    Ektaper = self.calc_top_taper(self.HEk, self.z, scalar=False)
    return tau_ave / self.f / self.rho * self.L * silltaper * Ektaper

  # --------------------------------------------------------------------------
  #  GM BVP boundary conditions
  # --------------------------------------------------------------------------
  def bc_GM(self, ya, yb):
    r"""
    Calculate the residuals of boundary conditions for the eddy-driven transport
    boundary value problem.

    Parameters
    ----------
    ya : ndarray
         Bottom boundary condition (values of y at deepest level). Units: Sv
    yb : ndarray
         Surface boundary condition (values of y at surface). Units: Sv

    Returns
    -------
    bc : ndarray
         If ``bvp_with_Ek`` is False, an array containing the bottom boundary
         condition ya[0] and surface boundary condition yb[0].
         If ``bvp_with_Ek`` is True, an array containing the residuals of the
         supplied boundary conditions and the value of the Ekman transport
         :math:`\Psi_{Ek}` at those boundaries.
    """

    if self.bvp_with_Ek:
      return np.array([
          ya[0] + self.Psi_Ek[0] * 1e6, yb[0] + self.Psi_Ek[-1] * 1e6
      ])
    else:
      return np.array([ya[0], yb[0]])

  # --------------------------------------------------------------------------
  #  GM eddy transport
  # --------------------------------------------------------------------------
  def calc_GM(self):
    r"""
    Compute the eddy (Gent & McWilliams) transport based on the meridionally
    averaged isopycnal slope.

    .. math::
      \begin{aligned}
      \Psi_{GM} &= K_{GM}\cdot s \\
      s(b) &\equiv \frac{z_B(b)}{L_y - y_{SO}(b)}
      \end{aligned}

    Where :math:`z_B(b)` is the depth of isopycnals of buoyancy class :math:`b`
    in the adjoining basin, and :math:`y_{SO}(b)` is the outcropping latitude
    of isopycnals of buoyancy class :math:`b` in the Southern Ocean, available
    via :meth:`TH_Psi_SO.ys`.

    Returns
    -------
    Psi_GM : ndarray
             Meridional average of the eddy transport at each vertical level
             of the Southern Ocean model (in m^3/s).
    """

    dy_atz = 0 * self.z
    eps = 0.1    # minimum dy (in meters) (to avoid division by 0)
    for ii in range(0, np.size(self.z)):
      dy_atz[ii] = max(self.y[-1] - self.ys(self.b(self.z[ii])), eps)

    bottaper = self.calc_bottom_taper(self.Htaperbot, self.z)
    toptaper = self.calc_top_taper(self.Htapertop, self.z)

    if self.c is not None:
      temp = make_func(
          self.KGM * self.z / dy_atz * self.L * toptaper * bottaper, self.z,
          'psiGM'
      )
      N2 = self.calc_N2()

      def ode(z, y):
        return np.vstack((y[1], N2(z) / self.c**2. * (y[0] - temp(z))))

      # Solve the boundary value problem
      res = integrate.solve_bvp(
          ode, self.bc_GM, self.z, np.zeros((2, np.size(self.z)))
      )
      # return solution interpolated onto original grid
      temp = res.sol(self.z)[0, :]
    else:
      temp = self.KGM * np.maximum(
          self.z / dy_atz, -self.smax
      ) * self.L * toptaper * bottaper

    # limit Psi_GM to -Psi_Ek on isopycnals that don't outcrop:
    idx = dy_atz > self.y[-1] - self.y[0]
    temp[idx] = np.maximum(temp[idx], -self.Psi_Ek[idx] * 1e6)
    return temp

  # --------------------------------------------------------------------------
  #  Solve residual SO overturning
  # --------------------------------------------------------------------------
  def solve(self):
    r"""
    Compute the residual overturning transport in the Southern Ocean.

    .. math::
      \Psi_{SO} = \Psi_{Ek} + \Psi_{GM}

    Returns
    -------
    Psi : ndarray
          Meridional average of the residual overturning transport at each
          vertical level of the Southern Ocean (in Sv).
    """

    self.Psi_Ek = self.calc_Ekman() / 1e6
    self.Psi_GM = self.calc_GM() / 1e6
    self.Psi = self.Psi_Ek + self.Psi_GM
    # Notice that the Psi at the bottom boundary is somewhat poorly defined,
    # and only used for plotting purposes, for which it makes sense to
    # simply set it to zero:
    self.Psi[0] = 0.

  # --------------------------------------------------------------------------
  #  Update buoyancy and surface buoyancy
  # --------------------------------------------------------------------------
  def update(self, col_basin=None, b=None, bs=None):
    r"""
    Update the vertical buoyancy profile and surface buoyancy, based on changes
    in the adjoining basin and/or in the surface boundary conditions.

    Parameters
    ----------
    col_basin : object, optional
        Column-like object (e.g. :class:`ThermohalineColumn`) providing the
        updated basin buoyancy profile, via its ``b`` attribute.
    b : float, function, or ndarray, optional
        Vertical buoyancy profile from the adjoining basin, on the north
        side of the ACC. Units: m/s^2
    bs : float, function, or ndarray, optional
         Surface level buoyancy boundary condition. Can be a constant,
         or an array or function in y. Units: m/s^2
    """

    if col_basin is not None:
      self.col_basin = col_basin

    if b is not None:
      self.b = make_func(b, self.z, 'b')
    elif (self.col_basin is not None) and hasattr(self.col_basin, 'b'):
      zb = np.asarray(self.col_basin.z)
      bb = np.asarray(self.col_basin.b)
      if np.allclose(zb, self.z):
        b_arg = bb
      else:
        def b_arg(zz, zb_=zb, bb_=bb):
          return np.interp(zz, zb_, bb_)
      self.b = make_func(b_arg, self.z, 'b')

    if bs is not None:
      self.bs = make_func(bs, self.y, 'bs')
