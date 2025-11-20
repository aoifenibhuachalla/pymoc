import numpy as np
from pymoc.utils import make_array


class SO_ML(object):
  r"""
  Southern Ocean Mixed Layer Model

  Instances of this class represent 1D representations of the meridional buoyancy structure
  in the Southern Ocean mixed layer, based on the solution to an advective-diffusive equation.
  Surface fluxes can be specified via a prescribed  fixed flux at the surface and/or
  a restoring buoyancy. Boundary conditions at the northern and southern boundaries
  of the Southern Ocean are determined internally based on the overturning streamfunction
  and buoyancy profile in the basin.


  ! Edits made 11 Nov 2025 to allow propagation of a passive tracer !

  """
  def __init__(
      self,
      y=None,    # grid (input)
      Ks=0.,    # hor. diffusivity (input)
      h=50.,    # ML depth (input)
      L=4e6,    # zonal width (input)
      surflux=0.,    # prescribed surface buoyancy flux (in m^2/s^3; input)
    # Notice that the first and last gridpoints are BCs and should have no flux
      rest_mask=0.,    # mask for surface restoring (1 where restoring is applied 0 elsewhere)
      b_rest=0.,    # surface buoyancy towards which we are restoring 
      v_pist=1.5 / 86400.,    # Piston velocity for restoring in SL (input)
      bs=0.0,    # Surface buoyancy (input, output)
      Psi_s=None,    # Overturning in the ML (output)
      tracer = 0.0
  ):
    r"""
    Parameters
    ----------

    y : ndarray
        Uniform meridional mixed layer grid. Units: m
    Ks : float
         Horizontal diffusivity in the mixed layer. Units: 
    h : float
        Mixed layer depth. Units: m
    L : float
        Zonal length of the mixed layer. Units: m
    surflux : float, ndarray, or function; optional
              Prescribed surface buoyancy flux. Units: m\ :sup:`2`/s\ :sup:`3`
    rest_mask : float, ndarray, or function; optional
                Surface restoring mask, prescribed as 1 where restoring is desired, and 0 elsewhere.
    b_rest : float, ndarray, or function; optional
             Prescribed surface buoyancy profile towards which mixed layer buoyancy is restored. Units: 
    v_pist : float; optional
             Prescribed piston velocity for buoyancy restoration. Units: m/s
    bs : float, ndarray, or function; optional
         Initial meridional surface buoyancy profile in the mixed layer. Units:
    Psi_s : ndarray; optional
         Initial overturning transport in the mixed layer. Units: Sv 

    """
    # initialize grid:
    if isinstance(y, np.ndarray):
      self.y = y
    else:
      raise TypeError('y needs to be numpy array providing (regular) grid')

    self.Ks = Ks
    self.h = h
    self.L = L
    self.surflux = make_array(surflux, self.y, 'surflux')
    self.rest_mask = make_array(rest_mask, self.y, 'rest_mask')
    self.b_rest = make_array(b_rest, self.y, 'b_rest')
    self.v_pist = v_pist
    self.Psi_s = Psi_s
    self.bs = make_array(bs, self.y, 'bs')
    self.tracer = make_array(tracer, self.y, 'tracer')

  # def solve_equi(self):
  #Solve for equilibrium solution given inputs
  # raise TypeError('This functionality is not yet implemented')

  # def set_boundary_conditions(self, b_basin, Psi_b, tracer_basin = None):
  #   r"""
  #   Set the surface boundary conditions at the southern boundary of the Southern
  #   Ocean mixed layer. If upwelling takes place at the boundary, set the buoyancy
  #   there to match the densest upwelled density class from the adjoining basin.
  #   Otherwisee, impose a no-flux boundary condition.

  #   Parameters
  #   ----------

  #   b_basin : ndarray
  #             The buoyancy profile in the adjoining basin. Units:
  #   Psi_b : ndarray
  #           The transport streamfunction in the Southern Ocean. Units: Sv

  #   """
  #   if self.Psi_s[1] > 0:
  #     # set buoyancy at southern boundary to buoyancy of densest upwelling water
  #     self.bs[0] = b_basin[np.argwhere(Psi_b > 0)[0][0]]
  #     if tracer_basin is not None:
  #         self.tracer[0] = tracer_basin[np.argwhere(Psi_b > 0) [0][0]]
  #   else:
  #     # no-flux BC
  #     self.bs[0] = self.bs[1]
  #     if tracer_basin is not None:
  #         self.tracer[0] = self.tracer[1]

  def set_boundary_conditions(self, b_basin, Psi_b, tracer_basin=None):
    """
    Set southern boundary BCs for buoyancy and tracer based on basin upwelling.
    If Psi_s indicates upwelling in the SO and there is upwelled water from the basin
    (Psi_b > 0 somewhere), set bs[0] and tracer[0] from the basin's densest upwelled class.
    Otherwise use no-flux (copy adjacent interior point).
    """
    # Guard Psi_s presence
    if self.Psi_s is None or len(self.Psi_s) < 2:
        # No overturning available: enforce no-flux BC (copy neighbor)
        self.bs[0] = self.bs[1]
        if tracer_basin is not None:
            self.tracer[0] = self.tracer[1]
        return  
    # find first positive basin-upwelling index (densest upwelled class)
    pos = np.argwhere(Psi_b > 0)
    if self.Psi_s[1] > 0 and pos.size > 0:
        idx = pos[0][0]
        self.bs[0] = b_basin[idx]
        if tracer_basin is not None:
            self.tracer[0] = tracer_basin[idx]
    else:
        # no-flux boundary condition: copy neighbor value
        self.bs[0] = self.bs[1]
        if tracer_basin is not None:
            self.tracer[0] = self.tracer[1]  
          
  # def set_boundary_conditions(self, b_basin, Psi_b, tracer_basin=None):  ## new !!
  #     # Guard Psi_s
  #     if self.Psi_s is None or len(self.Psi_s) < 2:
  #         # Fallback: no-flux at southern boundary
  #         self.bs[0] = self.bs[1]
  #         if tracer_basin is not None:
  #             self.tracer[0] = self.tracer[1]
  #         return

  #     pos = np.argwhere(Psi_b > 0)
  #     if self.Psi_s[1] > 0 and pos.size > 0:
  #         idx = pos[0][0]
  #         self.bs[0] = b_basin[idx]
  #         if tracer_basin is not None:
  #             self.tracer[0] = tracer_basin[idx]
  #     else:
  #         # no-flux BC
  #         self.bs[0] = self.bs[1]
  #         if tracer_basin is not None:
  #             self.tracer[0] = self.tracer[1]

  # def set_tracer_boundary_conditions(self, tracer_basin, Psi_b):
  #     if self.Psi_s is None or len(self.Psi_s) < 2:
  #         self.tracer[0] = self.tracer[1]
  #         return
  #     pos = np.argwhere(Psi_b > 0)
  #     if self.Psi_s[1] > 0 and pos.size > 0:
  #         idx = pos[0][0]
  #         self.tracer[0] = tracer_basin[idx]
  #     else:
  #         self.tracer[0] = self.tracer[1]



  def calc_advective_tendency(self, dy):
    r"""
    Compute the advective transport tendency in the Southern Ocean mixed layer:

    .. math::
        \begin{aligned}
        \left[\partial_tb_{SO}\right]_{adv} &= -v_{SO}^\dagger\partial_yb_{SO} \\
        v_{SO}^\dagger &= \frac{\Psi_{SO}}{h\cdot L_x}
        \end{aligned}

    Parameters
    ----------

    dy: float
        The horizontal grid spacing for the mixed layer model.

    Returns
    -------

    dbdb_ad: ndarray
             An array representing the advecting tendency at each grid point.

    """

    # Notice that the current implementation assumes an evenly spaced grid!
    if self.Psi_s is None or len(self.Psi_s) < 3:
        return 0.0 * self.y

    dbdt_ad = 0. * self.y
    indneg = self.Psi_s[1:-1] < 0.
    indpos = self.Psi_s[1:-1] > 0.
    dbdt_ad[1:-1][indneg] = -self.Psi_s[1:-1][indneg] * 1e6 * (
        self.bs[2:][indneg] - self.bs[1:-1][indneg]
    ) / self.h / self.L / dy
    dbdt_ad[1:-1][indpos] = -self.Psi_s[1:-1][indpos] * 1e6 * (
        self.bs[1:-1][indpos] - self.bs[:-2][indpos]
    ) / self.h / self.L / dy
    return dbdt_ad

# !!!!!!!!!!!!!!!!!!!!!!!!!!

  def calc_advective_tendency_tracer(self, dy):
      """
      Compute meridional advective tendency for tracer in the SO mixed layer.
      Uses the same upwind discretization as calc_advective_tendency for buoyancy.
      Returns tendency array of same length as self.y (zero if Psi_s not set).
      """
      if self.Psi_s is None or len(self.Psi_s) < 3:
          return 0.0 * self.y
  
      dtdt_ad = 0.0 * self.y
      indneg = self.Psi_s[1:-1] < 0.0
      indpos = self.Psi_s[1:-1] > 0.0
  
      # when Psi_s < 0: flow from north -> south, donor is northern neighbor (y+1 index)
      dtdt_ad[1:-1][indneg] = -self.Psi_s[1:-1][indneg] * 1e6 * (
          self.tracer[2:][indneg] - self.tracer[1:-1][indneg]
      ) / self.h / self.L / dy
  
      # when Psi_s > 0: flow from south -> north, donor is southern neighbor (y-1 index)
      dtdt_ad[1:-1][indpos] = -self.Psi_s[1:-1][indpos] * 1e6 * (
          self.tracer[1:-1][indpos] - self.tracer[:-2][indpos]
      ) / self.h / self.L / dy
  
      return dtdt_ad

  def apply_diffusion_tracer(self, dt, dy):
      """
      Apply implicit lateral diffusion to self.tracer over one timestep dt using
      the same FTCS implicit matrix as for buoyancy.
      This function updates self.tracer in-place.
      """
      # build stability number s and matrix U (reuse calc_diffusion_matrix)
      s = self.Ks * dt / (dy * dy)
      U = self.calc_diffusion_matrix(s)
  
      # right-hand side is current tracer (no source term here)
      rhs = self.tracer.copy()
  
      # impose BC consistency: U already sets first and last rows to Dirichlet identity
      # but ensure tracer at boundaries are consistent (no-flux treated as copy)
      rhs[0] = self.tracer[0]
      rhs[-1] = self.tracer[-1]
  
      # solve linear system U * q_new = rhs for q_new
      # use numpy.linalg.solve (matrix is small and tridiagonal-ish)
      q_new = np.linalg.solve(U, rhs)
  
      # update
      self.tracer = q_new


  def calc_diffusion_matrix(self, s):
    r"""
    Compute the forward time centered space (FTCS) coeffient matrix for the implicit diffusion scheme, based on
    the stability criterion:

    .. math::
        s = K_s\cdot\frac{dt}{dy^2}

    Paramaers
    ---------
    s: float
       Stability criterion for FTCS implicit solution.

    Returns
    -------
    U: ndarray
       Coefficient matrix representing the system of linear equations being solved by the implicit diffusion scheme.

    """
    U = (
        np.diag(-s / 2. * np.ones(len(self.y) - 1), -1) +
        np.diag((1+s) * np.ones(len(self.y)), 0) +
        np.diag(-s / 2. * np.ones(len(self.y) - 1), 1)
    )
    U[0, 0] = 1
    U[0, 1] = 0
    U[-1, -2] = 0
    U[-1, -1] = 1

    return U

  def calc_implicit_diffusion(self, dy, dt):
    r"""
    Compute the diffusive transport in Southern Ocean mixed layer:

    .. math::
        \left[\partial_tb_{SO}\right]_{di\hspace{-0.1em}f\hspace{-0.2em}f} = \partial_y\left(K_s\partial_yb_{SO}\right)

    via an implicit scheme and apply it to the surface buoyancy.

    Parameters
    ----------
    
    dy: float
        The horizontal grid spacing for the mixed layer model.
    dt: float
        The timestep duration for the mixed layer model.

    Returns
    -------

    bs: ndarray
        Updated surface buoyancy, with the implicit diffusion applied.
    """

    s = self.Ks * dt / dy**2
    U = self.calc_diffusion_matrix(s)
    V = self.calc_diffusion_matrix(-s)
    Uinv = np.linalg.inv(U)

    return np.dot(np.dot(Uinv, V), self.bs)

     ## !!!!!!!!!!!!!!!!!!!
  def calc_implicit_diffusion_tracer(self, dy, dt):
      """
      Implicit FTCS diffusion applied to tracer (same Ks, h, grid).
      Returns updated tracer array.
      """
      s = self.Ks * dt / dy**2
      U = self.calc_diffusion_matrix(s)
      V = self.calc_diffusion_matrix(-s)
  
      # Solve U * q_new = V * q_old, avoiding explicit inverse
      rhs = V.dot(self.tracer)
      q_new = np.linalg.solve(U, rhs)
      return q_new


  def advdiff(self, b_basin, Psi_b, dt, tracer_basin=None):
    r"""
    Compute and apply advective-diffusive transport to the Souther Ocean Mixed Layer model,
    based on conditions in the adjoining basin.

    Parameters
    ----------

    b_basin: ndarray
              The buoyancy profile in the adjoining basin. Units:
    Psi_b: ndarray
            The transport streamfunction in the Southern Ocean. Units: Sv
    dt: float
        The timestep duration for the mixed layer model.

    """
    # update surface buoyancy profile via advect. and diff

    # First we need to determine Psi at the surface in the ML:
    # The first line here is a hack to reduce problems with interpolation
    # due to finite SO resolution when PsiSO goes to zero at the bottom
    #due to non-outcropping isopycnals
    # If Psi_b is zero at non-outcropping isopycnals, it can end up
    # very near zero also at the last (non-boundary) gridpoint
    # at the surface as a result of the interpolation procedure
    # - this is unphysical since psi by definition only vanishes on isopycnals
    # that don't outcrop (the problem is that this vanishing here is a step
    # function which messes with the interpolation.)
    # For the purpose of the interpolation to the surface we therefore set psi
    # on non-outcropping isopycnals to the last non-zero value above
    Psi_mod = Psi_b.copy()
    # ind = np.nonzero(Psi_mod)[0][0]
    # Psi_mod[:ind] = Psi_mod[ind]
    nz = np.nonzero(Psi_mod)[0]
    if nz.size > 0:   ## new !!
        ind = nz[0]
        Psi_mod[:ind] = Psi_mod[ind]

    self.Psi_s = np.interp(self.bs, b_basin, Psi_mod)

    # The following makes sure that the circulation vanishes at latitudes
    # south of the densest surface buoyancy in case of non-monotonicity
    # of bs around Antarctica (a lense of lighter water around Antarctica
    # cannot connect to the channel and thus should have zero overturning.
    # Notice, however, that the model is not generally designed to properly
    # handle non-monotonic bs, so any such solution should treated with care)
    self.Psi_s[:np.argmin(self.bs)] = 0.
    self.Psi_s[
        0
    ] = 0.    # This value doesn't actually enter/matter, but zero overturning
    # at southern boundary makes more sense for diag purposes

    #set boundary conditions:
    self.set_boundary_conditions(b_basin, Psi_b)

    # Compute tendency due to surface b-flux
    dbdt_flux = self.surflux / self.h + self.rest_mask * self.v_pist / self.h * (
        self.b_rest - self.bs
    )

    # Compute advective tendency via upwind advection
    dy = self.y[1] - self.y[0]
    dbdt_ad = self.calc_advective_tendency(dy)

    # add tendencies from surface flux and advection
    self.bs = self.bs + dt * (dbdt_flux+dbdt_ad)

    #re-set southern boundary condition
    #(the above does not modify the boundary gridpoints,
    # but if we want to simulate no-flux bbc we need to adjust the boundary point accordingly):
    if self.Psi_s[1] <= 0:
      # no-flux BC
      self.bs[0] = self.bs[1]

    # Do implicit diffusion:
    self.bs = self.calc_implicit_diffusion(dy, dt)

    #re-set southern boundary condition:
    # this final re-set is just to pass back a state where boundary point is consistent with bcs
    # (preferable e.g. for computation of streamfunction)
      
    self.set_boundary_conditions(b_basin, Psi_b)

    
    if tracer_basin is not None:
        # Set tracer BCs consistent with current overturning
        self.set_boundary_conditions(b_basin, Psi_b, tracer_basin=tracer_basin)

        dtdt_ad = self.calc_advective_tendency_tracer(dy)
        self.tracer = self.tracer + dt * dtdt_ad

        if self.Psi_s is not None and len(self.Psi_s) > 1 and self.Psi_s[1] <= 0:
            self.tracer[0] = self.tracer[1]
    
        self.tracer = self.calc_implicit_diffusion_tracer(dy, dt)
    
        self.set_boundary_conditions(b_basin, Psi_b, tracer_basin=tracer_basin)

    

  # def timestep(self, b_basin=None, Psi_b=None, dt=1., tracer_basin = None):
  #   r"""
  #   Integrate the mixed layer buoyancy profile for one timestep.

  #   Parameters
  #   ----------

  #   b_basin: ndarray
  #             The buoyancy profile in the adjoining basin. Units:
  #   Psi_b: ndarray
  #           The transport streamfunction in the Southern Ocean. Units: Sv
  #   dt: float
  #       The timestep duration for the mixed layer model.

  #   """

  #   #Integrate buoyancy profile evolution for one time-step
  #   if not isinstance(b_basin, np.ndarray):
  #     raise TypeError(
  #         'b_basin needs to be numpy array providing buoyancy levels in basin'
  #     )

  #   if not isinstance(Psi_b, np.ndarray):
  #     raise TypeError(
  #         'Psi_b needs to be numpy array providing overturning at buoyancy levels given by b_basin'
  #     )

  #   self.advdiff(b_basin=b_basin, Psi_b=Psi_b, dt=dt)

  #   if tracer_basin is not None:  ## new !!!
  #       self.advdiff(b_basin=b_basin, Psi_b=Psi_b, dt=dt, tracer_basin = tracer_basin)

  # new !! (12 Nov) 
  def tracer_inventory(self):  
      """
      Return total tracer inventory in the SO mixed layer.
      Inventory = sum(tracer * cell_area * h) with cell_area = L * dy
      """
      dy = self.y[1] - self.y[0]
      cell_area = self.L * dy
      total = np.sum(self.tracer * cell_area * self.h)
      return total

 ## new !! (12 Nov) 
  def tracer_on_grid(self):
     """
     Return tracer array on the mixed-layer grid for coupling to basin modules.
     Keeps interface same as bs for buoyancy coupling.
     """
     return self.tracer.copy()  


  def timestep(self, b_basin=None, Psi_b=None, dt=1., tracer_basin=None):  ## new !! 
      if not isinstance(b_basin, np.ndarray):
          raise TypeError('b_basin needs to be numpy array providing buoyancy levels in basin')
      if not isinstance(Psi_b, np.ndarray):
          raise TypeError('Psi_b needs to be numpy array providing overturning at buoyancy levels given by b_basin')

      self.advdiff(b_basin=b_basin, Psi_b=Psi_b, dt=dt, tracer_basin=tracer_basin)
          


## new !! (12 Nov)
  def timestep_tracer(self, dt, dy, include_advection=True, include_diffusion=True):
      """
      Advance tracer in the Southern Ocean ML one timestep dt.
      - include_advection: apply advective tendency (upwind) if True
      - include_diffusion: apply implicit lateral diffusion if True
      """
      # 1) advective tendency
      if include_advection:
          adv_tend = self.calc_advective_tendency_tracer(dy)
          self.tracer += dt * adv_tend
  
      # 2) implicit diffusion
      if include_diffusion:
          self.apply_diffusion_tracer(dt, dy)
  
      # 3) boundary consistency (ensure BC points reflect any coupling decisions)
      # For no-flux behavior, copy neighbor for boundary unless set elsewhere
      self.tracer[0] = self.tracer[0]  # leave as set by set_boundary_conditions or coupling
      self.tracer[-1] = self.tracer[-1]  # typically no change (end BC)

