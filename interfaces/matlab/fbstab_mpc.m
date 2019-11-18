

classdef fbstab_mpc < handle

methods(Access = public)
  function o = fbstab_mpc(horizon_length,num_states,num_inputs,num_constraints)

    o.N = horizon_length;
    o.nx = num_states;
    o.nu = num_inputs;
    o.nc = num_constraints;

    o.nz = (o.N+1)*(o.nx+o.nu);
    o.nl = (o.N+1)*o.nx;
    o.nv = (o.N+1)*o.nc;
  end

  function [nz,nl,nv] = problem_size(o)
    nz = o.nz;
    nl = o.nl;
    nv = o.nv;
  end

  function [x,out] = solve(o,data,x0)
  end

  function opts = default_options()
  end

end % public methods

methods(Access = public) % These will be private later
  function ValidateProblemData(o,data)

  end

  function ValidateInitialGuess(o,x0)
  end

end % public properties

properties(Access = private)
  N;
  nx;
  nu;
  nc;

  nz;
  nl;
  nv;

end

end