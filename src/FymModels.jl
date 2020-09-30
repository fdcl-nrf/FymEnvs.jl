module FymModels
"""
module FymModels

This module provides dynamical equations of (famous) flight systems.
"""


import FymEnvs: dyn

include("FymCore.jl")
using .FymCore

export FlightDynamicalSystems
export system, dyn
# flight dynamical systems list
export F16LinearLateral

############ FlightDynamicalSystems ############
abstract type FlightDynamicalSystems end

function dyn(fds::FlightDynamicalSystems)
    return fds.dyn
end

function system(fds::FlightDynamicalSystems)
    return fds.system
end


mutable struct F16LinearLateral <: FlightDynamicalSystems
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
    system::FymSystem
    A
    B
    C
    dyn
    F16LinearLateral(; kwargs...) = init!(new(); kwargs...)
end

function init!(fds::F16LinearLateral;
               initial_state=[1, 0, 0, 0, 0, 0, 0])
    fds.system = BaseSystem(initial_state=initial_state)
    fds.A = [
             [-0.322, 0.064, 0.0364, -0.9917, 0.0003, 0.0008, 0],
             [0, 0, 1, 0.0037, 0, 0, 0],
             [-30.6492, 0, -3.6784, 0.6646, -0.7333, 0.1315, 0],
             [8.5396, 0, -0.0254, -0.4764, -0.0319, -0.062, 0],
             [0, 0, 0, 0, -20.2, 0, 0],
             [0, 0, 0, 0, 0, -20.2, 0],
             [0, 0, 0, 57.2958, 0, 0, -1]
            ]
    fds.B = [
             [0, 0],
             [0, 0],
             [0, 0],
             [0, 0],
             [20.2, 0],
             [0, 20.2],
             [0, 0]
            ]
    fds.C = [
             [0, 0, 0, 57.2958, 0, 0, -1],
             [0, 0, 57.2958, 0, 0, 0, 0],
             [57.2958, 0, 0, 0, 0, 0, 0],
             [0, 57.2958, 0, 0, 0, 0, 0]
            ]
    fds.dyn = function(x, u)
        return fds.A * x + fds.B * u
    end
    return fds
end



end
