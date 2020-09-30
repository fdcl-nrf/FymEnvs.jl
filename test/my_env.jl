module My_Env


using FymEnvs
using LinearAlgebra

export my_env, reset!, step!, observe
export my_sys


function my_env(; controller=my_controller, x0=zeros(3), kwargs...)
    env = BaseEnv(; kwargs...)
    my_systems = Dict("my_sys" => BaseSystem(initial_state=x0))
    systems!(env, my_systems)
    dyn!(env, set_dyn)

    params = env.params
    params["controller"] = controller
    return env
end

params(env) = env.params
control(env, x) = params(env)["controller"](x)

function set_dyn(env, t)  # corresponding to set_dot of the original FymEnvs
    my_sys = systems(env)["my_sys"]
    x = state(my_sys)
    u = control(env, x)
    A = Matrix(I, 3, 3)
    B = Matrix(I, 3, 3)
    dot!(my_sys, A*x + B*u)
end

function FymEnvs.reset!(env)
    # needs to be extended
    reset!(env)
end

observe(env) = observe_array(env)

function step!(env)
    t = time(env.clock)
    my_sys = systems(env)["my_sys"]
    x = state(my_sys)
    u = control(env, x)
    info = Dict("time" => t, "state" => x, "input" => u)

    update!(env)
    next_obs = state(my_sys)
    reward = zeros(1)
    done = time_over(env.clock)
    return next_obs, reward, done, info
end

my_controller(x; K=3*Matrix(I, 3, 3)) = -K * x




end
