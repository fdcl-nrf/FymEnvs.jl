module My_Env


using FymEnvs
const FDS = FlightDynamicalSystems

using LinearAlgebra
using Parameters

export my_env, reset!, observe
export my_controller


############ FDS ############
@with_kw mutable struct MySystem <: FDS
    initial_state = zeros(3)
    A = Matrix(I, 3, 3)
    B = Matrix(I, 3, 3)
end

function deriv(mysys::MySystem)
    deriv = function(x, u)
        return mysys.A*x + mysys.B*u
    end
    return deriv
end


function my_env(; x0=zeros(3), kwargs...)
    mysys = MySystem(initial_state=x0)
    function set_dyn(env, t; deriv=deriv(mysys))
        # corresponding to set_dot of the original FymEnvs
        sys = env.systems["my_sys"]
        x = sys.state
        u = my_controller(x)
        sys.dot = deriv(x, u)
    end
    function step(env)
        t = time(env.clock)
        sys = env.systems["my_sys"]
        x = sys.state
        u = my_controller(x)
        info = Dict("time" => t, "state" => x, "input" => u)

        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        return next_obs, reward, done, info
    end

    systems = Dict("my_sys" => system(mysys))
    env = BaseEnv(; kwargs...)
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    return env
end


function FymEnvs.reset!(env)
    # extend it if necessary
    reset!(env)
end

observe(env) = observe_array(env)

my_controller(x; K=3*Matrix(I, 3, 3)) = -K * x




end
