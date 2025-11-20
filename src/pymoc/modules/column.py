import numpy as np
from scipy import integrate
from pymoc.utils import make_func, make_array, check_numpy_version

class Column(object):
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

  ! Edits made 11 Nov 2025 to allow for propagation of a passive tracer !
  """
  def __init__(
      self,
      z=None,    # grid (input)
      kappa=None,    # diffusivity profile (input)
      bs=0.025,    # surface buoyancy bound. cond (input)
      bbot=0.0,    # bottom buoyancy boundary condition (input)  
      bzbot=None,    # bottom strat. as alternative boundary condition (input) 
      b=0.0,    # Buoyancy profile (input, output)
      Area=None,    # Horizontal area (can be function of depth)
      N2min=1e-7, # Minimum strat. for conv adjustment
      tracer = 0.0   # passive tracer possible 
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
    """

    # initialize grid:
    if isinstance(z, np.ndarray) and len(z) > 0:
      self.z = z
    else:
      raise TypeError('z needs to be numpy array providing grid levels')

    self.kappa = make_func(kappa, self.z, 'kappa')
    self.Area = make_func(Area, self.z, 'Area')

    self.bs = bs
    self.bbot = bbot
    self.bzbot = bzbot

    self.N2min = N2min

    self.b = make_array(b, self.z, 'b')
    self.tracer = make_array(tracer, self.z, 'tracer')   ## new!!

    if check_numpy_version():
      self.bz = np.gradient(self.b, z)
      #self.tracerz = np.gradient(self.tracer, z)
        
    else:
      self.bz = 0. * z    # notice that this is just for initialization of ode solver

    

    ## new !! 


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
              self.tracer[-1] += amount
          else:
              A_surf = self.Area(self.z[-1])
              if len(self.z) > 1:
                  dz_surf = self.z[-1] - self.z[-2]
              else:
                  raise ValueError("Cannot infer layer thickness from a single grid level.")
              V_surf = A_surf * dz_surf
              self.tracer[-1] += amount / V_surf
          return
  
      zmin, zmax = depth_range
      zlow = min(zmin, zmax)
      zhigh = max(zmin, zmax)
      mask = (self.z >= zlow) & (self.z <= zhigh)
      if not np.any(mask):
          return
  
      if as_concentration:
          self.tracer[mask] += amount
      else:
          idxs = np.where(mask)[0]
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
          self.tracer[idxs] += amount * W








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

    if self.bzbot is None:
      return np.array([ya[0] - self.bbot, yb[0] - self.bs])
    else:
      return np.array([ya[1] - self.bzbot, yb[0] - self.bs])

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
    sol_init[0, :] = self.b
    sol_init[1, :] = self.bz
    res = integrate.solve_bvp(self.ode, self.bc, self.z, sol_init)
    # interpolate solution for b and db/dz onto original grid
    self.b = res.sol(self.z)[0, :]
    self.bz = res.sol(self.z)[1, :]

  def vertadvdiff(self, wA, dt, do_conv=False):
    r"""
    Calculate and apply the forcing from advection and diffusion on the vertical buoyancy
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
      self.b[-1]=self.bs;
    self.b[0] = (self.bbot if self.bzbot is None
                  else self.b[1] - self.bzbot * dz[0])

    bz = (self.b[1:] - self.b[:-1]) / dz
    bz_up = bz[1:]
    bz_down = bz[:-1]
    bzz = (bz_up-bz_down) / (0.5 * (dz[1:] + dz[:-1]))

    #upwind advection:
    weff = wA - self.dAkappa_dz(self.z)
    bz = bz_down
    bz[weff[1:-1] < 0] = bz_up[weff[1:-1] < 0]

    db_dt = (
        -weff[1:-1] * bz / self.Area(self.z[1:-1]) +
        self.kappa(self.z[1:-1]) * bzz
    )
    self.b[1:-1] = self.b[1:-1] + dt*db_dt

    # !!!!!!!!!!!!!!!!!!!!
  def vertadvdiff_tracer(self, wA, dt, do_conv=False):
      """
      Vertical advection-diffusion for tracer, mirroring buoyancy transport.
      Uses upwind advection based on effective vertical transport.
      """
      wA = make_array(wA, self.z, 'wA')
      dz = self.z[1:] - self.z[:-1]
  
      tz = (self.tracer[1:] - self.tracer[:-1]) / dz
      tz_up = tz[1:]    # gradient above interface
      tz_down = tz[:-1] # gradient below interface
  
      tzz = (tz_up - tz_down) / (0.5 * (dz[1:] + dz[:-1]))
  
      # (same as buoyancy)
      weff = wA - self.dAkappa_dz(self.z)
  
      tz_adv = tz_down.copy()
      mask = (weff[1:-1] < 0.0)
      tz_adv[mask] = tz_up[mask]
  
      # Tendency 
      dtracer_dt = (
          -weff[1:-1] * tz_adv / self.Area(self.z[1:-1]) +
          self.kappa(self.z[1:-1]) * tzz
      )
  
      self.tracer[1:-1] += dt * dtracer_dt
  
      #  non-negativity clamp
      self.tracer = np.maximum(self.tracer, 0.0)


  # def vertadvdiff_tracer(self, wA, dt):
  #     """
  #     Conservative vertical tracer update over dt using interface transports wA (m^3/s).
  #     wA is expected as an array on the z-grid giving interface transports consistent with AMOC outputs:
  #       - length Nz (if given as interface transports matching Psi_iso works),
  #       - we'll interpret wA as per-layer interface transport; adjust if your wA is defined differently.
  #     This routine computes advective fluxes (upwind) and diffusive fluxes (Fickian) at interfaces,
  #     then updates layer concentrations by the divergence of fluxes divided by layer volumes.
  #     """
  #     # ensure arrays on z
  #     z = self.z
  #     Nz = len(z)
  #     dz = self._layer_thicknesses()       # layer thickness array length Nz (bottom..top)
  #     # layer volumes
  #     A_layer = self.Area(z)
  #     V = A_layer * dz
  
  #     # make wA array sized Nz-1 interfaces. If wA length equals Nz, assume it denotes interface transports aligned with Psi differences.
  #     w = np.asarray(wA)
  #     if w.size == Nz:
  #         # assume transport defined per layer; approximate interface transports by averaging neighbours
  #         w_if = 0.5 * (w[:-1] + w[1:])
  #     elif w.size == Nz - 1:
  #         w_if = w.copy()
  #     else:
  #         # fallback: broadcast scalar
  #         w_if = np.ones(Nz - 1) * float(w)
  
  #     # compute advective interface tracer using upwind donor from column tracer
  #     # tracer array indexed 0..Nz-1 bottom->top; interface i is between layer i and i+1
  #     trac = self.tracer.copy()
  #     t_if_adv = np.zeros(Nz - 1)
  #     for i in range(Nz - 1):
  #         if w_if[i] > 0:
  #             # flow upward across interface (from layer i -> i+1)
  #             t_if_adv[i] = trac[i]
  #         else:
  #             # flow downward across interface (from layer i+1 -> i)
  #             t_if_adv[i] = trac[i + 1]
  
  #     # advective flux (volume per second times concentration) at each interface
  #     F_adv = w_if * t_if_adv   # m^3/s * concentration => mass per second (concentration units * m^3/s)
  
  #     # diffusive flux: F_diff = -A_interface * kappa * (C_top - C_bot)/dz_interface
  #     # approximate interface area as A_layer averaged
  #     A_if = 0.5 * (A_layer[:-1] + A_layer[1:])
  #     # interface separation dz_if: distance between layer centers
  #     dz_if = 0.5 * (dz[:-1] + dz[1:])
  #     # tracer gradient top - bot across interface:
  #     grad = (trac[1:] - trac[:-1]) / dz_if
  #     F_diff = - A_if * self.kappa * grad    # units: concentration * m^2/s * m^-1 * m^2 => concentration * m^3/s
  
  #     # total flux at interfaces (positive upward if we adopt w_if positive upward)
  #     F_if = F_adv + F_diff
  
  #     # update layers by divergence of interface fluxes: dM/dt = F_{i-1/2} - F_{i+1/2}
  #     # note indexing for bottom layer i=0 uses F_if[0] as top interface, bottom boundary flux assumed zero
  #     dCdt = np.zeros_like(trac)
  #     # bottom layer (i=0)
  #     dCdt[0] = (0.0 - F_if[0]) / V[0]
  #     # interior layers
  #     for i in range(1, Nz - 1):
  #         dCdt[i] = (F_if[i - 1] - F_if[i]) / V[i]
  #     # top layer (i = Nz-1)
  #     dCdt[-1] = (F_if[-1] - 0.0) / V[-1]
  
  #     # update concentrations
  #     self.tracer = self.tracer + dCdt * dt




  def convect(self):
    r"""
    Carry out downward convective adustment of the vertical buoyancy profile to
    the minimum stratification, :math:`N^2_{m\!i\!n}`. The current implimentation 
    assumes a fixed buoyancy at the bottom of the convective region (to be interpreted
    as the minimum surfcae buoyancy).

    """

    # do convective adjustment to minimum strat N2min
    # Notice that this parameterization currently only handles convection
    # from the top down which is the only case we really encounter here...
    # The BC is applied such that we are fixing b at bottom of the convective layer
    ind=self.b>self.bs
    if ind.any():
      # z_conv is top-most non-convetive layer (set to bottom of the ocean if all convecting):
      zconv= np.max(self.z[np.invert(ind)]) if np.invert(ind).any() else self.z[0]
      self.b[ind]=self.bs+self.N2min*(self.z[ind]-zconv)
    else:
      # if no convection simply set bs as upper BC  
      self.b[-1]=self.bs  
        
    # in a previous version we instead fixed the actual surface buoyancy;
    # the code for that approach is here:
    #ind = self.b > self.bs + self.N2min * self.z
    #self.b[ind] = self.bs + self.N2min * self.z[ind]
    # Below is an energy conserving version that could be used
    # for model formulations without fixed surface b. But for fixed surface
    # b, the simpler version above is preferable as it deals better with long time steps
    # (for infinitesimal time-step and fixed surface b, the two are equivalent)
    # Moreover, this version does not currently include adjustment to finite strat.
    # dz=self.z[1:]-self.z[:-1];
    # dz=np.append(dz,dz[-1]);dz=np.insert(dz,0,dz[0])
    # dz=0.5*(dz[1:]+dz[0:-1]);
    # self.b[ind]=(np.mean(self.b[ind]*dz[ind]*self.Area(self.z[ind]))
    #            /np.mean(dz[ind]*self.Area(self.z[ind])) )



  def convect_tracer(self):
      """
      Mix tracer uniformly over the convective layer detected by buoyancy convect().
      Conserves tracer mass within that layer.
      """
      ind = self.b > self.bs
      if not ind.any():
          return
  
      conv_idx = np.where(ind)[0]
  
      dz_all = np.empty_like(self.z, dtype=float)
      dz_all[1:-1] = 0.5 * (self.z[2:] - self.z[:-2])
      dz_all[0] = self.z[1] - self.z[0]
      dz_all[-1] = self.z[-1] - self.z[-2]
  
      V = self.Area(self.z[conv_idx]) * dz_all[conv_idx]
      mass = np.sum(self.tracer[conv_idx] * V)
  
      if np.sum(V) > 0.0:
          q_uniform = mass / np.sum(V)
          self.tracer[conv_idx] = q_uniform


  # def horadv(self, vdx_in, b_in, dt):
  #   r"""
  #   Carry out horizon buoyancy advection into the column model from an adjoining model,
  #   for the timestepping solution. This function implements an upwind advection scheme.

  #   Parameters
  #   ----------

  #   vdx_in : float or ndarray
  #            Total advective transport per unit height into the column for the timestepping
  #            solution. Positive values indicate transport into the column. Units: m\ :sup:`2`/s
  #   b_in : float or ndarray
  #          Buoyancy vales from the adjoining module for the timestepping solution. Units: m/s\ :sup:`2`
  #   dt : int
  #        Numerical timestep over which solution are iterated. Units: s

  #   """

  #   vdx_in = make_array(vdx_in, self.z, 'vdx_in')
  #   b_in = make_array(b_in, self.z, 'b_in')

  #   adv_idx = vdx_in > 0.0
  #   db = b_in - self.b

  #   self.b[adv_idx] = self.b[adv_idx] + dt * vdx_in[adv_idx] * db[
  #       adv_idx] / self.Area(self.z[adv_idx])
      
  # def horadv_tracer(self, vdx_in, tracer_in, dt):
  #     """
  #     Upwind horizontal advection for tracer: inflow-only update.
  #     vdx_in > 0 means inflow into the column; donor is tracer_in.
  #     """
  #     vdx_in = make_array(vdx_in, self.z, 'vdx_in')
  #     tracer_in = make_array(tracer_in, self.z, 'tracer_in')
  
  #     adv_idx = vdx_in > 0.0
  #     dtracer = tracer_in - self.tracer
  
  #     self.tracer[adv_idx] += dt * vdx_in[adv_idx] * dtracer[adv_idx] /  self.Area(self.z[adv_idx])



  def _layer_thicknesses(self):
      dz = np.empty_like(self.z, dtype=float)
      dz[1:-1] = 0.5 * (self.z[2:] - self.z[:-2])
      dz[0] = self.z[1] - self.z[0]
      dz[-1] = self.z[-1] - self.z[-2]
      return np.abs(dz)
  
  # def horadv(self, vdx_in, b_in, dt):
  #     vdx_in = make_array(vdx_in, self.z, 'vdx_in')
  #     b_in = make_array(b_in, self.z, 'b_in')
  #     adv_idx = vdx_in > 0.0
  #     db = b_in - self.b
  #     dz = self._layer_thicknesses()
  #     A = self.Area(self.z)
  #     self.b[adv_idx] += dt * vdx_in[adv_idx] * db[adv_idx] / (A[adv_idx] * dz[adv_idx])

  def horadv(self, vdx_in, b_in, dt):
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
      b_in = make_array(b_in, self.z, 'b_in')
  
      adv_idx = vdx_in 
      db = b_in - self.b

      self.b[adv_idx] = self.b[adv_idx] + dt * vdx_in[adv_idx] * db[
        adv_idx] / self.Area(self.z[adv_idx])

  def horadv_tracer(self, vdx_in, tracer_in, dt):
    vdx_in = make_array(vdx_in, self.z, 'vdx_in')
    tracer_in = make_array(tracer_in, self.z, 'tracer_in')

    inflow = vdx_in > 0
    outflow = vdx_in < 0

    # Inflow: add tracer using neighbour value
    dtr = tracer_in - self.tracer
    self.tracer[inflow] = (
        self.tracer[inflow]
        + dt * vdx_in[inflow] * dtr[inflow] / self.Area(self.z[inflow])
    )

    # Outflow: remove tracer proportional to local value
    self.tracer[outflow] = (
        self.tracer[outflow]
        + dt * vdx_in[outflow] * self.tracer[outflow] / self.Area(self.z[outflow])
    )


  # def horadv_tracer(self, vdx_in, tracer_in, dt):
  #     """
  #     Upwind horizontal advection for tracer: inflow-only update per layer.
  #     vdx_in > 0 means inflow into this column from neighbor holding tracer_in.
  #     """
        
  #     vdx_in = make_array(vdx_in, self.z, 'vdx_in')
  #     tracer_in = make_array(tracer_in, self.z, 'tracer_in')
  
  #     adv_idx = vdx_in >0
  #     dtracer = tracer_in - self.tracer

  #     self.tracer[adv_idx] = self.tracer[adv_idx] + dt * vdx_in[adv_idx] * dtracer[
  #       adv_idx] / self.Area(self.z[adv_idx])


  # def horadv_tracer(self, vdx_in, tracer_in, dt):
  #     vdx_in = make_array(vdx_in, self.z, 'vdx_in')
  #     tracer_in = make_array(tracer_in, self.z, 'tracer_in')
  
  #     A = self.Area(self.z)
  #     dz = self._layer_thicknesses()
  
  #     # horizontal fluxes across interfaces
  #     flux_in = vdx_in * tracer_in          # inflow flux
  #     flux_out = np.roll(vdx_in, -1) * np.roll(self.tracer, -1)  # outflow from each layer
  
  #     # assume zero flux out bottommost cell
  #     flux_out[-1] = 0.0
  
  #     # divergence (in - out)
  #     dFdz = flux_in - flux_out
  
  #     self.tracer += dt * dFdz / (A * dz)


  # def apply_tracer_streamfunction_flux(self, tracer_flux, dt):
  #     """
  #     Apply tracer tendency from overturning streamfunction flux divergence.
  #     tracer_flux: array of shape (len(z)) giving net tracer flux into each layer (units: tracer * m^3/s)
  #     dt: timestep in seconds
  #     """
  #     dz = self._layer_thicknesses()
  #     V = self.Area(self.z) * dz
  #     self.tracer += dt * tracer_flux / V
  #     self.tracer = np.maximum(self.tracer, 0.0)  # optional clamp




  def timestep(self, wA=0., dt=1., do_conv=False, vdx_in=None, b_in=None, tracer_in=None, tracer_injection=None):   # new 12 Nov
      if do_conv:
          self.convect()
          self.convect_tracer()
  
      self.vertadvdiff(wA=wA, dt=dt, do_conv=do_conv)
      self.vertadvdiff_tracer(wA=wA, dt=dt, do_conv = do_conv)
  
      if vdx_in is not None:
          if b_in is not None:
              self.horadv(vdx_in=vdx_in, b_in=b_in, dt=dt)
          else:
              raise TypeError('b_in is needed if vdx_in is provided')
  
          if tracer_in is not None:
              self.horadv_tracer(vdx_in=vdx_in, tracer_in=tracer_in, dt=dt)
        
  # def timestep(self, wA=0., dt=1., do_conv=False, vdx_in=None, b_in=None, tracer_in=None, tracer_injection=None):
  #   r"""
  #   Carry out one timestep integration for the buoyancy profile, accounting
  #   for advective, diffusive, and convective effects.

  #   Parameters
  #   ----------

  #   wA : float or ndarray
  #        Area integrated velocity profile for the timestepping solution. Units: m\ :sup:`3`/s
  #   dt : int
  #        Numerical timestep over which solution are iterated. Units: s
  #   do_conv : logical
  #             Whether to carry out convective adjustment during model integration.
  #   vdx_in : float or ndarray
  #            Total advective transport per unit height into the column for the timestepping
  #            solution. Positive values indicate transport into the column. Units: m\ :sup:`2`/s
  #   b_in : float or ndarray
  #          Buoyancy vales from the adjoining module for the timestepping solution. Units: m/s\ :sup:`2`

  #   """
  #   if do_conv:
  #     # do convection: (optional)
  #     self.convect()
    
  #   # do vertical advection and diffusion
  #   self.vertadvdiff(wA=wA, dt=dt, do_conv=do_conv)
  #   self.vertadvdiff_tracer(wA = wA, dt = dt)
    
  #   if vdx_in is not None:
  #     # do horizontal advection: (optional)
  #     if b_in is not None:
  #       self.horadv(vdx_in=vdx_in, b_in=b_in, dt=dt)
  #     else:
  #       raise TypeError('b_in is needed if vdx_in is provided')
  #     if tracer_in is not None: ## new !!
  #         self.horadv_tracer(vdx_in = vdx_in, tracer_in=tracer_in, dt=dt)
          
  #   if tracer_injection is not None:   ## new !!
  #       self.add_tracer(**tracer_injection)


  def tracer_inventory(self):
      
      dz_all = np.empty_like(self.z, dtype = float)
      dz_all[1:-1] = 0.5 * (self.z[2:] - self.z[:-2])
      dz_all[0] = self.z[1] - self.z[0]
      dz_all[-1] = self.z[-1] - self.z[-2]
      V = self.Area(self.z) * dz_all
      
      return np.sum(self.tracer * V)