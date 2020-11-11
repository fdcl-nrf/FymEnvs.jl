module TestEnvs

using LinearAlgebra
using Parameters
using FymEnvs

export TestEnv


mutable struct TestEnv <: FymEnv
    env::BaseEnv
    controller
    TestEnv(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(fym::TestEnv;
               controller=LinearController(), x0=ones(3), kwargs...)
    fym.env = BaseEnv(; name="custom", kwargs...)
    fym.controller = controller

    function set_dyn(env, t)
        sys = env.systems["test_sys"]
        x = sys.state
        u = get(fym.controller, x)
        A = Matrix(I, 3, 3)
        B = Matrix(I, 3, 3)
        sys.dot = A*x + B*u
    end
    function step(env; action=nothing)
        t = time(env.clock)
        sys = env.systems["test_sys"]
        x = sys.state
        u = get(fym.controller, x)
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


@with_kw mutable struct LinearController
    K = 3*Matrix(I, 3, 3)
end
Base.get(ctrl::LinearController, x) = -ctrl.K * x


end
