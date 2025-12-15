import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from pymoc.utils import make_func, make_array


class TH_Psi_Thermwind(object):
  r"""
  Thermal Wind Closure for Thermohaline Columns

  Instances of this class represent the overturning circulation between two
  vertical columns, given **buoyancy** profiles in those columns. The buoyancy
  profiles are typically diagnosed from temperature and salinity via a linear
  equation of state in a :class:`ThermohalineColumn`, but can also be provided
  directly.

  The model assumes a thermal-wind based equation for the overturning
  circulation as in Nikurashin and Vallis (2012):

  .. math::
    \partial_{zz}\left(\Psi\right) = f^{-1} (b_2 - b_1),

  where :math:`b_1(z)` and :math:`b_2(z)` are the buoyancy profiles in the
  southern basin and northern deep-water formation region, respectively.

  This equation is solved subject to the boundary conditions:

  .. math::
    \Psi(0) = \Psi(-H) = 0.

  An upwind isopycnal mapping is then used to compute the isopycnal overturning
  transport.

  Parameters
  ----------
  f : float, optional
      Coriolis parameter. Units: s^-1
  z : ndarray, optional
      Vertical depth levels of overturning grid. Units: m. If omitted and
      ``col1`` is provided, the grid is taken from ``col1.z``.
  sol_init : ndarray, optional
      Initial guess at the solution to the thermal wind overturning
      streamfunction (for the BVP solver). Shape (2, nz), representing
      :math:`(\Psi, \partial_z \Psi)`.
  col1 : object, optional
      First column-like object, typically a :class:`ThermohalineColumn`, from
      which the buoyancy profile in the southern basin is taken. Must provide
      attributes ``z`` and ``b``.
  col2 : object, optional
      Second column-like object, typically a :class:`ThermohalineColumn`, from
      which the buoyancy profile in the northern region is taken. Must provide
      attributes ``z`` and ``b``.
  b1 : float, function, or ndarray, optional
      Vertical buoyancy profile from the southern basin. Units: m/s^2
  b2 : float, function, or ndarray, optional
      Vertical buoyancy profile from the northern basin, representing the
      deep-water formation region. Units: m/s^2

  Notes
  -----
  If both ``col1``/``col2`` and ``b1``/``b2`` are provided, the explicit
  ``b1``/``b2`` arguments take precedence.
  """

  def __init__(
      self,
      f=1.2e-4,    # Coriolis parameter (input)
      z=None,      # grid (input)
      sol_init=None,    # Initial conditions for ODE solver (input)
      col1=None,        # ThermohalineColumn-like object for basin
      col2=None,        # ThermohalineColumn-like object for deep water region
      b1=None,          # Buoyancy in the basin (input, output)
      b2=None,          # Buoyancy in the deep water formation region (input, output)
  ):

    self.f = f

    # Determine grid:
    if isinstance(z, np.ndarray):
      self.z = z
    elif (z is None) and (col1 is not None) and hasattr(col1, 'z'):
      # take grid from first column if not provided explicitly
      self.z = np.asarray(col1.z)
    else:
      raise TypeError('z needs to be numpy array providing grid levels, '
                      'or col1 must provide a .z attribute.')

    nz = np.size(self.z)

    # Initialize buoyancy fields, possibly from columns:
    # b1: southern basin
    if b1 is not None:
      self.b1 = make_func(b1, self.z, 'b1')
    elif (col1 is not None) and hasattr(col1, 'b'):
      self.b1 = make_func(col1.b, self.z, 'b1')
    else:
      # default: zero buoyancy profile
      self.b1 = make_func(0.0, self.z, 'b1')

    # b2: northern region
    if b2 is not None:
      self.b2 = make_func(b2, self.z, 'b2')
    elif (col2 is not None) and hasattr(col2, 'b'):
      self.b2 = make_func(col2.b, self.z, 'b2')
    else:
      # default: zero buoyancy profile
      self.b2 = make_func(0.0, self.z, 'b2')

    # Store column references (optional, for later update convenience)
    self.col1 = col1
    self.col2 = col2

    # Set initial conditions for BVP solver
    if sol_init is None:
      self.sol_init = np.zeros((2, nz))
    else:
      self.sol_init = sol_init

    # Placeholder for streamfunction in depth space
    self.Psi = np.zeros_like(self.z)

  # --------------------------------------------------------------------------
  #  Boundary conditions for BVP
  # --------------------------------------------------------------------------
  def bc(self, ya, yb):
    r"""
    Calculate the residuals of boundary conditions for the thermal wind closure
    boundary value problem.

    Parameters
    ----------
    ya : ndarray
         Bottom boundary condition (values of :math:`y` at the deepest level).
    yb : ndarray
         Surface boundary condition (values of :math:`y` at the surface).

    Returns
    -------
    bc : ndarray
         An array containing the residuals of the imposed boundary conditions:

         * :math:`\Psi(-H) = 0`
         * :math:`\Psi(0) = 0`
    """

    return np.array([ya[0], yb[0]])

  # --------------------------------------------------------------------------
  #  ODE system for BVP
  # --------------------------------------------------------------------------
  def ode(self, z, y):
    r"""
    Generate the ordinary differential equation for the thermal wind overturning
    streamfunction, to be solved as a boundary value problem:

    .. math::

        \partial_{zz}\left(\Psi\right) = f^{-1} (b_2 - b_1).

    Parameters
    ----------
    z : ndarray
        Vertical depth levels of column grid on which to solve the ODE. Units: m
    y : ndarray
        Initial values for the streamfunction and its vertical gradient, i.e.:

        .. math::
            y = (\Psi, \partial_z \Psi).

    Returns
    -------
    ode : ndarray
          A vertically oriented array, containing the system of linear equations:

          .. math::
            \begin{aligned}
            \partial_z y_1 &= y_2 \\
            \partial_z y_2 &= \frac{b_2(z) - b_1(z)}{f}
            \end{aligned}
    """
    return np.vstack((y[1], 1.0 / self.f * (self.b2(z) - self.b1(z))))

  # --------------------------------------------------------------------------
  #  Solve thermal-wind BVP
  # --------------------------------------------------------------------------
  def solve(self):
    r"""
    Solve for the thermal wind overturning streamfunction as a boundary value
    problem based on the system of equations defined in
    :meth:`TH_Psi_Thermwind.ode`.

    The resulting streamfunction :math:`\Psi(z)` is stored in ``self.Psi`` in
    Sverdrups.
    """

    # Note: The solution to this BVP is a relatively straightforward integral;
    # it would probably be faster to just code it up that way. We use solve_bvp
    # for consistency with the original implementation.
    res = integrate.solve_bvp(self.ode, self.bc, self.z, self.sol_init)
    # interpolate solution for overturning circulation onto original grid
    # and change units to Sv:
    self.Psi = res.sol(self.z)[0, :] / 1e6

  # --------------------------------------------------------------------------
  #  Isopycnal mapping of overturning
  # --------------------------------------------------------------------------
  def Psib(self, nb=500):
    r"""
    Remap the overturning streamfunction from physical depth space into
    isopycnal space:

    .. math::
      \Psi^b(b) = \int_{-H}^0 \partial_z\Psi(z)\,\mathcal{H}[b - b_{\text{up}}(z)]\,dz,

    by computing upstream (upwind) density classes:

    .. math::
      \begin{aligned}
      b_{\text{up}}(z) =
      \begin{cases}
        b_2(z), & \partial_z\Psi(z)  > 0 \\
        b_1(z), & \partial_z\Psi(z)  < 0
      \end{cases}
      \end{aligned}

    where :math:`b_1(z)` is the buoyancy profile in the southern basin,
    :math:`b_2(z)` is the buoyancy profile in the northern region, and
    :math:`\mathcal{H}` is the Heaviside step function.

    Parameters
    ----------
    nb : int, optional
         Number of upstream density classes into which the streamfunction is
         to be remapped.

    Returns
    -------
    psib : ndarray
           An array representing the values of the overturning streamfunction
           in each upwind density class.
    """
    # Map overturning into isopycnal space:
    b1 = make_array(self.b1, self.z, 'b1')
    b2 = make_array(self.b2, self.z, 'b2')

    bmin = min(np.min(b1), np.min(b2))
    bmax = max(np.max(b1), np.max(b2))
    self.bgrid = np.linspace(bmin, bmax, nb)

    # udydz = -dPsi/dz (note the sign convention vs. original code)
    udydz = -(self.Psi[1:] - self.Psi[:-1])
    psib = 0.0 * self.bgrid

    # upstream buoyancy at bottom/top of each layer
    bup_bot = b1[:-1].copy()
    bup_top = b1[1:].copy()
    idx = udydz < 0
    bup_bot[idx] = b2[:-1][idx]
    bup_top[idx] = b2[1:][idx]

    for i in range(0, len(self.bgrid)):
      # fraction of each layer included in this buoyancy class
      mask = np.clip((bup_top - self.bgrid[i]) / (bup_top - bup_bot), 0.0, 1.0)
      psib[i] = np.sum(mask * udydz)

    return psib

  # --------------------------------------------------------------------------
  #  Isopycnal-depth mapping back to columns
  # --------------------------------------------------------------------------
  def Psibz(self, nb=500):
    r"""
    Remap the overturning streamfunction onto the native isopycnal-depth space
    of the columns in the southern basin and northern region.

    Parameters
    ----------
    nb : int, optional
         Number of upstream density classes into which the streamfunction is
         to be remapped in the intermediate :meth:`TH_Psi_Thermwind.Psib` step.

    Returns
    -------
    psibz : list of ndarrays
            A list with two elements:

            * ``psibz[0]`` : streamfunction at each depth level in the southern basin.
            * ``psibz[1]`` : streamfunction at each depth level in the northern region.
    """
    # Map isopycnal overturning into isopycnal space:
    psib = self.Psib(nb)

    # This does a linear interpolation in b:
    return [
        np.interp(self.b1(self.z), self.bgrid, psib),
        np.interp(self.b2(self.z), self.bgrid, psib)
    ]

    # Alternative approach (commented out), interpolating in depth:
    # z1_of_bgrid = np.interp(self.bgrid, self.b1(self.z), self.z)
    # z2_of_bgrid = np.interp(self.bgrid, self.b2(self.z), self.z)
    # return [np.interp(self.z, z1_of_bgrid, psib),
    #         np.interp(self.z, z2_of_bgrid, psib)]

  # --------------------------------------------------------------------------
  #  Update buoyancy profiles
  # --------------------------------------------------------------------------
  def update(self, col1=None, col2=None, b1=None, b2=None):
    r"""
    Update the vertical buoyancy profiles from the southern basin and northern
    region, either from new buoyancy arrays/functions or from updated column
    objects.

    Parameters
    ----------
    col1 : object, optional
        First column-like object, typically a :class:`ThermohalineColumn`,
        whose ``b`` attribute is used for the southern basin buoyancy.
    col2 : object, optional
        Second column-like object, typically a :class:`ThermohalineColumn`,
        whose ``b`` attribute is used for the northern region buoyancy.
    b1 : float, function, or ndarray, optional
        Vertical buoyancy profile from the southern basin. Units: m/s^2
    b2 : float, function, or ndarray, optional
        Vertical buoyancy profile from the northern basin. Units: m/s^2
    """
    # Update stored column references if provided
    if col1 is not None:
      self.col1 = col1
    if col2 is not None:
      self.col2 = col2

    # Update buoyancy profiles
    if b1 is not None:
      self.b1 = make_func(b1, self.z, 'b1')
    elif (self.col1 is not None) and hasattr(self.col1, 'b'):
      self.b1 = make_func(self.col1.b, self.z, 'b1')

    if b2 is not None:
      self.b2 = make_func(b2, self.z, 'b2')
    elif (self.col2 is not None) and hasattr(self.col2, 'b'):
      self.b2 = make_func(self.col2.b, self.z, 'b2')
