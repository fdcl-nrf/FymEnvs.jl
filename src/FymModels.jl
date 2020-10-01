module FymModels
"""
module FymModels

This module provides dynamical equations of (famous) flight systems.

Usually, you would need to fill keyword argument `initial_state`.

[Usage]
f16 = F16LinearLateral(initial_state=ones(7))
systems = Dict("f16" => system(f16))
# Note: typeof(system(fds::FlightDynamicalSystems)) == BaseSystem
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


using Reexport
using Debugger
using Parameters

include("FymCore.jl")
@reexport using .FymCore

export FlightDynamicalSystems
export system, deriv

# flight dynamical systems list
export F16LinearLateral, GlidingVehicle3DOF

############ FlightDynamicalSystems ############
abstract type FlightDynamicalSystems end


@with_kw mutable struct GlidingVehicle3DOF <: FlightDynamicalSystems
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
end

function deriv(gv3::GlidingVehicle3DOF)
    # TODO: add wind
    deriv = function(state, input)
        x, y, z, V, gamma, chi = state
        CL, phi = input
        CD = gv3.CD0 + gv3.CD1^2

        dxdt = V *cos(gamma) * cos(chi)
        dydt = V * cos(gamma) * sin(chi)
        dzdt = - V * sin(gamma)
        dVdt = - gv3._term * V^2 * CD - gv3.g * sin(gamma)
        dgammadt = (gv3._term * V * CL * cos(phi)
                    - gv3.g * cos(gamma) / V)
        dchidt = gv3._term * V / cos(gamma) * CL * sin(phi)
        return [dxdt, dydt, dzdt, dVdt, dgammadt, dchidt]
    end
    return deriv
end

function system(gv3::GlidingVehicle3DOF)
    initial_state = gv3.initial_state
    name = gv3.name
    return BaseSystem(initial_state=initial_state, name=name)
end


@with_kw mutable struct F16LinearLateral <: FlightDynamicalSystems
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
end

function deriv(f16::F16LinearLateral)
    deriv = function(x, u)
        return f16.A * x + f16.B * u
    end
    return deriv
end

function system(f16::GlidingVehicle3DOF)
    initial_state = f16.initial_state
    name = f16.name
    return BaseSystem(initial_state=initial_state, name=name)
end



end
