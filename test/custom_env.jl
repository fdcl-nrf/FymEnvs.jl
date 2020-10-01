module TestEnvs

using LinearAlgebra
using FymEnvs

export test_env
export TestEnv


mutable struct TestEnv <: FymEnv
    env::BaseEnv
    controller
    TestEnv(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(fym::TestEnv;
               controller=linear_controller, x0=ones(3), kwargs...)
    fym.env = BaseEnv(; kwargs...)
    fym.controller = controller

    function set_dyn(env, t; fym::TestEnv=fym)
        sys = env.systems["test_sys"]
        x = sys.state
        u = fym.controller(x)
        A = Matrix(I, 3, 3)
        B = Matrix(I, 3, 3)
        sys.dot = A*x + B*u
    end
    function step(env; action=nothing, fym::TestEnv=fym)
        t = time(env.clock)
        sys = env.systems["test_sys"]
        x = sys.state
        u = fym.controller(x)
        info = Dict("time" => t, "state" => x, "input" => u)

        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        return next_obs, reward, done, info
    end

    env = fym.env
    systems = Dict("test_sys" => BaseSystem(initial_state=x0))
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    return fym
end


function linear_controller(x; K=3*Matrix(I, 3, 3))
    return -K*x
end


end
