

import numpy as np
from scipy import integrate
from pymoc.utils import make_func, make_array, check_numpy_version


class Qolumn(object):
  r"""
  Vertical Advection-Diffusion Column Model

  Instances of this class represent 1D representations of buoyancy in a water
  column governed by vertical advection and diffusion. The velocity
  profile is required as an input. The script can either compute the equilibrium
  buoyancy profile for a given vertical velocity profile and boundary conditions
  or compute the tendency and perform a time-step of given length.
  BCs have to be fixed buoyancy at the top and either fixed b or db/dz at the bottom
  The time-stepping version can also handle horizontal advection
  into the column. This is, however, not (yet) implemented for the equilibrium solver
  """
  def __init__(
      self,
      z=None,    # grid (input)
      kappa=None,    # diffusivity profile (input)
      
      qs = 0.0,  # surface tracer boundary condition  (input)
      qbot = 0.0, # bottom tracer boundary condition (input) 
      qzbot = None,   # alt. boundary condition 
      q = 0.0,   # tracer profile (input, output) 
      
      Area=None,    # Horizontal area (can be function of depth)
      N2min=1e-7    # Minimum strat. for conv adjustment
  ):
    r"""
    Parameters
    ----------

    z : ndarray
        Vertical depth levels of column grid. Units: m
    kappa : float, function, or ndarray
            Vertical diffusivity profile. Units: m\ :sup:`2`/s
    bs : float
         Surface level buoyancy boundary condition. Units: m/s\ :sup:`2`
    bbot : float; optional
           Bottom level buoyancy boundary condition. Units: m/s\ :sup:`2`
    bzbot : float; optional
            Bottom level buoyancy stratification. Can be used as an alternative to **bbot**. Units: s\ :sup:`-2`
    b : float, function, or ndarray
        Initial vertical buoyancy profile. Recalculated on model run. Units: m/s
    Area : float, function, or ndarray
           Horizontal area of basin. Units: m\ :sup:`2`
    N2min : float; optional
            Minimum stratification for convective adjustment. Units: s\ :sup:`-1`

    Q is for tracer -> equivalent to buoyancy variable. Passive tracer should be treated similarly to buoyancy and should be equivalently transported arounf the system. 
    """

    # initialize grid:
    if isinstance(z, np.ndarray) and len(z) > 0:
      self.z = z
    else:
      raise TypeError('z needs to be numpy array providing grid levels')

    self.kappa = make_func(kappa, self.z, 'kappa')
    self.Area = make_func(Area, self.z, 'Area')


    self.qs = qs
    self.qbot = qbot
    self.qzbot = qzbot

    self.N2min = N2min

    self.q = make_array(q, self.z, 'q')


    if check_numpy_version():
      self.qz = np.gradient(self.q, z)
    else:
      self.qz = 0. * z    # notice that this is just for initialization of ode solver
    

  def Akappa(self, z):
    r"""
    Compute the area integrated diffusivity :math:`A\kappa`
    at depth(s) z.

    Parameters
    ----------

    z : float or ndarray
        Vertical depth level(s) at which to retrieve the integrated diffusivity.
    
    Returns
    -------
    AKappa : float or ndarray
             If z is a number, a number corresponding to the integrated diffusivity :math:`A\kappa` at that depth.
             If z is an ndarray, an ndarray where each entry corresponds to the integrated diffusivity :math:`A\kappa`
             at the z value with the same index.

    """

    return self.Area(z) * self.kappa(z)

  def dAkappa_dz(self, z):
    r"""
    Compute the area integrated diffusivity gradient :math:`\partial_z\left(A\kappa\right)`
    at depth(s) z.

    Parameters
    ----------

    z : float or ndarray
        Vertical depth level(s) at which to retrieve the integrated diffusivity gradient.

    Returns
    -------

    dAkappa_dz : float or ndarray
                 If z is a number, a number corresponding to the ther vertical gradients
                 in the integrated diffusivity :math:`\partial_zA\kappa` at that depth.
                 If z is an ndarray, an ndarray where each entry corresponds to the vertical gradient
                 in the integrated diffusivity :math:`\partial_zA\kappa` at the z value with the same index.

    """

    if not check_numpy_version():
      raise ImportError(
          'You need NumPy version 1.13.0 or later. Please upgrade your NumPy libary.'
      )
    return np.gradient(self.Akappa(z), z)

  def bc(self, ya, yb):
    r"""
    Calculate the residuals of boundary conditions for the advective-diffusive
    boundary value problem.

    Parameters
    ----------

    ya : ndarray
         Bottom boundary condition. Units: m/s\ :sup:`2`
    yb : ndarray
         Surface boundary condition. Units: m/s\ :sup:`2`

    Returns
    -------

    bc : ndarray
         If the bottom buoyancy stratification is defined as a boundary condition,
         an array containing the residuals of the imposed and calculated bottom buoyancy
         stratification and surface buoyancy.
         If the bottom buoyancy stratification is undefined as a boundary condition,
         an array containing the residuals of the imposed and calculated bottom buoyancy
         and surface buoyancy.

    """


    if self.qzbot is None:
      return np.array([ya[0] - self.qbot, yb[0] - self.qs])
    else:
      return np.array([ya[1] - self.qzbot, yb[0] - self.qs])

  def ode(self, z, y):
    r"""
    Generate the ordinary differential equation for the equilibrium buoyancy profile,
    to be solved as a boundary value problem:

    :math:`\partial_tb\left(z\right)=-w^\dagger\partial_zb+\partial_z\left(\kappa_{e\!f\!f}\partial_zb\right)`

    Parameters
    ----------

    z : ndarray
        Vertical depth levels of column grid on which to solve the ode. Units: m
    y : ndarray
        Initial values for buoyancy and buoyancy gradient profiles.

    Returns
    -------
    ode : ndarray
          A vertically oriented array, containing the system of linear equations:

          .. math::
            \begin{aligned}
            \partial_zy_1 &= y_2 \\
            \partial_zy_2 &= wA - \partial_zy_2\cdot\frac{\partial_zA\kappa}{A\kappa}
            \end{aligned}

    """

    return np.vstack(
        (y[1], (self.wA(z) - self.dAkappa_dz(z)) / self.Akappa(z) * y[1])
    )

  def solve_equi(self, wA):
    r"""
    Solve for the equilibrium buoyancy profile, given a specified vertical
    velocity profile, and pre-set surface and bottom boundary conditions, based
    on the system of equations defined by :meth:`pymoc.modules.Column.ode`.

    Parameters
    ----------

    wA : ndarray
         Area integrated velocity profile for the equilibrium solution. Units: m\ :sup:`3`/s

    """

    self.wA = make_func(wA, self.z, 'w')
    sol_init = np.zeros((2, np.size(self.z)))

    sol_init[0, :] = self.q
    sol_init[1, :] = self.qz
    res = integrate.solve_bvp(self.ode, self.bc, self.z, sol_init)
    # interpolate solution for b and db/dz onto original grid
    self.q = res.sol(self.z)[0, :]
    self.qz = res.sol(self.z)[1, :]

    

  def vertadvdiff(self, wA, dt, do_conv=False):
    r"""
    Calculate and apply the forcing from advection and diffusion on the vertical tracer
    profile, for the timestepping solution. This function implements an upwind advection
    scheme.

    Parameters
    ----------

    wA : float or ndarray
         Area integrated velocity profile for the timestepping solution. Units: m\ :sup:`3`/s
    dt : int
         Numerical timestep over which solution are iterated. Units: s

    """

    wA = make_array(wA, self.z, 'wA')
    dz = self.z[1:] - self.z[:-1]

    # apply boundary conditions:
    if not do_conv: # if we use convection, upper BC is already applied there
      self.q[-1]=self.qs;
    self.q[0] = (self.qbot if self.qzbot is None
                  else self.q[1] - self.qzbot * dz[0])

    qz = (self.q[1:] - self.q[:-1]) / dz
    qz_up = qz[1:]
    qz_down = qz[:-1]
    qzz = (qz_up-qz_down) / (0.5 * (dz[1:] + dz[:-1]))

    #upwind advection:
    weff = wA - self.dAkappa_dz(self.z)
    qz = qz_down
    qz[weff[1:-1] < 0] = qz_up[weff[1:-1] < 0]

    dq_dt = (
        -weff[1:-1] * qz / self.Area(self.z[1:-1]) +
        self.kappa(self.z[1:-1]) * qzz
    )
    self.q[1:-1] = self.q[1:-1] + dt*dq_dt


  def convect(self):
    r"""
    Carry out downward convective adustment of the vertical tracer profile to
    the minimum stratification, :math:`N^2_{m\!i\!n}`. The current implimentation 
    assumes a fixed buoyancy at the bottom of the convective region (to be interpreted
    as the minimum surfcae buoyancy).

    """

    # do convective adjustment to minimum strat N2min
    # Notice that this parameterization currently only handles convection
    # from the top down which is the only case we really encounter here...
    # The BC is applied such that we are fixing b at bottom of the convective layer
    ind=self.q>self.qs
    if ind.any():
      # z_conv is top-most non-convetive layer (set to bottom of the ocean if all convecting):
      zconv= np.max(self.z[np.invert(ind)]) if np.invert(ind).any() else self.z[0]
      self.q[ind]=self.qs+self.N2min*(self.z[ind]-zconv)
    else:
      # if no convection simply set bs as upper BC  
      self.q[-1]=self.qs  
        



  def horadv(self, vdx_in, q_in, dt):
    r"""
    Carry out horizon buoyancy advection into the column model from an adjoining model,
    for the timestepping solution. This function implements an upwind advection scheme.

    Parameters
    ----------

    vdx_in : float or ndarray
             Total advective transport per unit height into the column for the timestepping
             solution. Positive values indicate transport into the column. Units: m\ :sup:`2`/s
    b_in : float or ndarray
           Buoyancy vales from the adjoining module for the timestepping solution. Units: m/s\ :sup:`2`
    dt : int
         Numerical timestep over which solution are iterated. Units: s

    """

    vdx_in = make_array(vdx_in, self.z, 'vdx_in')
    q_in = make_array(q_in, self.z, 'q_in')

    adv_idx = vdx_in > 0.0
    dq = q_in - self.q

    self.q[adv_idx] = self.q[adv_idx] + dt * vdx_in[adv_idx] * dq[
        adv_idx] / self.Area(self.z[adv_idx])

  def add_tracer(self, amount, depth_range=None, as_concentration=True):
      """
      Add tracer to the column. Parameters
      ----------
      amount : float
        If as_concentration=True, amount is a concentration increment.
        If as_concentration=False, amount is total tracer quantity to distribute.
      depth_range : tuple (zmin, zmax), optional
        Depth interval over which to apply the addition. If None, apply at the surface.
        Use the same z sign convention as the model (surface is the maximum z value).
      as_concentration : bool
        Treat 'amount' as concentration increment (True) or total quantity (False).
  
      
      """
      # Surface layer is the last index in the column arrays
      if depth_range is None:
          if as_concentration:
              self.q[-1] += amount
          else:
              A_surf = self.Area(self.z[-1])
              if len(self.z) > 1:
                  dz_surf = self.z[-1] - self.z[-2]
              else:
                  raise ValueError("Cannot infer layer thickness from a single grid level.")
              V_surf = A_surf * dz_surf
              self.q[-1] += amount / V_surf
          return
  
      zmin, zmax = depth_range
      # allow users to give depth_range with zmin <= zmax or zmin >= zmax
      zlow = min(zmin, zmax)
      zhigh = max(zmin, zmax)
      mask = (self.z >= zlow) & (self.z <= zhigh)
      if not np.any(mask):
          return
  
      if as_concentration:
          self.q[mask] += amount
      else:
          idxs = np.where(mask)[0]
          # robust layer thickness for each index
          dz = np.empty_like(idxs, dtype=float)
          for k, i in enumerate(idxs):
              if i == 0:
                  dz[k] = self.z[1] - self.z[0]
              elif i == len(self.z) - 1:
                  dz[k] = self.z[-1] - self.z[-2]
              else:
                  dz[k] = 0.5 * (self.z[i+1] - self.z[i-1])
          A = self.Area(self.z[idxs])
          V = A * dz
          W = V / V.sum()
          self.q[idxs] += amount * W

  def timestep(self, wA=0., dt=1., do_conv=False, vdx_in=None, q_in = None):
    r"""
    Carry out one timestep integration for the buoyancy profile, accounting
    for advective, diffusive, and convective effects.

    Parameters
    ----------

    wA : float or ndarray
         Area integrated velocity profile for the timestepping solution. Units: m\ :sup:`3`/s
    dt : int
         Numerical timestep over which solution are iterated. Units: s
    do_conv : logical
              Whether to carry out convective adjustment during model integration.
    vdx_in : float or ndarray
             Total advective transport per unit height into the column for the timestepping
             solution. Positive values indicate transport into the column. Units: m\ :sup:`2`/s
    b_in : float or ndarray
           Buoyancy vales from the adjoining module for the timestepping solution. Units: m/s\ :sup:`2`

    """
    if do_conv:
      # do convection: (optional)
      self.convect()
        
    # do vertical advection and diffusion
    self.vertadvdiff(wA=wA, dt=dt, do_conv=do_conv)
    
    if vdx_in is not None:

          # do horizontal advection: (optional)
      if q_in is not None:
        self.horadv(vdx_in=vdx_in, q_in=q_in, dt=dt)
      else:
        raise TypeError('q_in is needed if vdx_in is provided')