"""
module FymModels

This module provides environments and systems of famous flight systems,
a.k.a. FymEnv and FymSystem, respectively.

[Usage]
- FymEnv

- FymSystem

f16 = F16LinearLateral(initial_state=ones(7))
systems = Dict("f16" => system(f16))
# Note: typeof(system(fds::FymSystem)) == BaseSystem
function step(env)
    update!(env)
    next_obs = sys.state
    done = time_over(env.clock)
    return next_obs, zeros(1), done, Dict()
end
function set_dyn(env, t; deriv=deriv(f16))
    sys = env.systems["f16"]
    x = sys.state
    u = zeros(2)
    sys.dot = deriv(x, u)
end

env = BaseEnv()
systems!(env, systems)
dyn!(env, set_dyn)
step!(env, step)
reset!(env)

⋮

"""
module FymModels


using Reexport
# using Debugger
using Parameters

include("FymCore.jl")
@reexport using .FymCore

export FymEnv, sys
export FymSystem

# FymSystem list
export F16LinearLateral, GlidingVehicle3DOF


############ FymEnv ############
# Rule of the type `FymEnv`
# 1) your custom env should be a subtype of `FymEnv`:
    # CustomEnv <: FymEnv == true
# 2) your custom env should have a field `env`, whose type is BaseEnv:
    # typeof(CustomEnv.env) == BaseEnv
abstract type FymEnv end

function FymCore.sys(fym::FymEnv, name)
    return sys(fym.env, name)
end


############ FymSystem ############
abstract type FymSystem end
"""
Reference:
    TBD (borrowed from the original fym, but notation is different)
Note:
    state = [x, y, z, V, gamma, chi]
        x, y, z: NED position [m]
        V: (ground) speed [m/s]
        gamma: vertical flight path angle [rad]
        chi: horizontal flight path angle [rad]
    input = [CL, phi]
        CL: lift coefficient
        phi: bank angle [rad]
"""
@with_kw mutable struct GlidingVehicle3DOF <: FymSystem
    g = 9.80665
    rho = 1.2215
    m = 8.5
    S = 0.65
    b = 3.44
    CD0 = 0.033
    CD1 = 0.017
    name = "aircraft"
    _term = 0.5 * (rho * S) / m

    initial_state = [0, 0, -5.0, 13.0, 0, 0]
    system = BaseSystem(initial_state=initial_state, name=name)
    deriv = function(state, input)
        # TODO: add wind
        x, y, z, V, gamma, chi = state
        CL, phi = input
        CD = CD0 + CD1^2

        dxdt = V *cos(gamma) * cos(chi)
        dydt = V * cos(gamma) * sin(chi)
        dzdt = - V * sin(gamma)
        dVdt = - _term * V^2 * CD - g * sin(gamma)
        dgammadt = (_term * V * CL * cos(phi)
                    - g * cos(gamma) / V)
        dchidt = _term * V / cos(gamma) * CL * sin(phi)
        return [dxdt, dydt, dzdt, dVdt, dgammadt, dchidt]
    end
end


"""
Reference:
    B. L. Stevens et al. "Aircraft Control and Simulation", 3/e, 2016
    Example 5.3-1: LQR Design for F-16 Lateral Regulator
Dynamics:
    x_dot = Ax + Bu
State:
    x = [beta, phi, p, r, del_a, del_r, x_w]
    beta, phi: [rad], p, r: [rad/s], del_a, del_r: [deg]
Control input:
    u = [u_a, u_r]  (aileron and rudder servo inputs, [deg])
"""
@with_kw mutable struct F16LinearLateral <: FymSystem
    A = [
         [-0.322 0.064 0.0364 -0.9917 0.0003 0.0008 0];
         [0 0 1 0.0037 0 0 0];
         [-30.6492 0 -3.6784 0.6646 -0.7333 0.1315 0];
         [8.5396 0 -0.0254 -0.4764 -0.0319 -0.062 0];
         [0 0 0 0 -20.2 0 0];
         [0 0 0 0 0 -20.2 0];
         [0 0 0 57.2958 0 0 -1]
        ]
    B = [
         [0 0];
         [0 0];
         [0 0];
         [0 0];
         [20.2 0];
         [0 20.2];
         [0 0]
        ]
    C = [
         [0 0 0 57.2958 0 0 -1];
         [0 0 57.2958 0 0 0 0];
         [57.2958 0 0 0 0 0 0];
         [0 57.2958 0 0 0 0 0]
        ]
    name = "f16"

    initial_state = [1.0, 0, 0, 0, 0, 0, 0]
    system = BaseSystem(initial_state=initial_state, name=name)
    deriv = function(x, u)
        return A * x + B * u
    end
end


end
