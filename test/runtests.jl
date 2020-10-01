using FymEnvs
using LinearAlgebra

using Plots
ENV["GKSwstype"]="nul"  # do not show plot
using Debugger

using Revise; includet("custom_env.jl")  # to avoid conflict
using .TestEnvs


function print_msg(test_name)
    println(">"^6*" "*test_name*" "*"<"^6)
end


function test_Fym()
    print_msg("FymEnvs")
    function set_dyn(env, t)
        # corresponding to `set_dot` of the original fym
        # you can use any names in this package
        sys = env.systems["sys"]
        x = sys.state
        A = Matrix(I, 3, 3)
        sys.dot = -A * x
    end
    function step(env)
        t = time(env.clock)
        sys = env.systems["sys"]
        x = sys.state
        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        info = Dict(
                    "time" => t,
                    "state" => x,
                   )
        return next_obs, reward, done, info
    end

    x0 = collect(1:3)
    systems = Dict("sys" => BaseSystem(initial_state=x0, name="3d_sys"))
    log_dir = "data/test"
    file_name = "fym.h5"
    logger = Logger(log_dir=log_dir, file_name=file_name)
    env = BaseEnv(max_t=100.00, logger=logger, name="test_env")
    systems!(env, systems)  # set systems; required
    dyn!(env, set_dyn)  # set dynamics; required
    step!(env, step)  # set step; required

    reset!(env)  # reset env; required before propagation
    obs = observe_flat(env)
    i = 0
    @time while true
        render(env)  # not mendatory; would make simulator slow
        next_obs, reward, done, info = env.step()
        obs = next_obs
        i += 1
        if done
            break
        end
    end
    close!(env)
    data = load(env.logger.path)
    @show env
    @show size(data["state"]["sys"])
end

function test_largescale_env()
    print_msg("large-scale envs (nested env)")
    function set_dyn(env, t, p=0.0)
        x = env.systems["sys"].state
        y = env.systems["sys2"].state
        A = Matrix(I, 25, 25)
        B = Matrix(I, 25, 25)
        env.systems["sys"].dot = -p * A * x
        env.systems["sys2"].dot = -B * y
        env.systems["env0"].systems["sys"].dot = -p * A * x
        env.systems["env0"].systems["sys2"].dot = -B * y
    end
    function set_dyn0(env, t, p=0.0)  # extend
        x = env.systems["sys"].state
        y = env.systems["sys2"].state
        A = Matrix(I, 25, 25)
        B = Matrix(I, 25, 25)
        env.systems["sys"].dot = -p * A * x
        env.systems["sys2"].dot = -B * y
    end
    function step(env, action=nothing)
        t = time(env.clock)
        x = env.systems["sys"].state
        y = env.systems["sys2"].state
        env0x = env.systems["env0"].systems["sys"].state
        env0y = env.systems["env0"].systems["sys2"].state
        update!(env)
        next_obs = env.systems["sys"].state
        reward = zeros(1)
        done = time_over(env.clock)
        info = Dict("time" => t, "state" => x, "state2" => y,
                    "env0state" => env0x, "env0state2" => env0y)
        return next_obs, reward, done, info
    end

    x0 = collect(1:25)
    y0 = ones(25)
    systems0 = Dict(
                   "sys" => BaseSystem(initial_state=x0),
                   "sys2" => BaseSystem(initial_state=y0),
                  )
    env0 = BaseEnv(systems=systems0, dyn=set_dyn0)
    systems = Dict(
                   "sys" => BaseSystem(initial_state=x0),
                   "sys2" => BaseSystem(initial_state=y0),
                   "env0" => env0,
                  )
    log_dir = "data/test"
    file_name = "largescale.h5"
    logger = Logger(log_dir=log_dir, file_name=file_name)
    env = BaseEnv(max_t=100.00, logger=logger)
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    # functions
    reset!(env)  # reset!
    # simulation
    obs = observe_flat(env)
    i = 0
    @time while true
        next_obs, reward, done, info = env.step()
        obs = next_obs
        i += 1
        if done
            break
        end
    end
    close!(env)
    data = load(env.logger.path)
    state = data["state"]["sys"]
    state2 = data["state"]["sys2"]
    env0state = data["state"]["env0"]["sys"]
    env0state2 = data["state"]["env0"]["sys2"]
    t = data["time"]
    p = plot(t, state)
    p2 = plot(t, state2)
    env0p = plot(t, env0state)
    env0p2 = plot(t, env0state2)
    savefig(p, joinpath(log_dir, "state.pdf"))
    savefig(p2, joinpath(log_dir, "state2.pdf"))
    savefig(env0p, joinpath(log_dir, "env0state.pdf"))
    savefig(env0p2, joinpath(log_dir, "env0state2.pdf"))
end

function test_custom_env()
    env = test_env()
    log_dir = "data/test"
    file_name = "custom.h5"
    reset!(env)
    _sample(env, nothing, log_dir, file_name)
    data = load(joinpath(log_dir, file_name))
    for name in ["state", "input"]
        p = plot(data["time"], data[name])
        savefig(p, joinpath(log_dir, "custom_"*name*".pdf"))
    end
end

function test_custom_fym()
    fym = TestEnv()

    env = fym.env
    log_dir = "data/test"
    file_name = "custom.h5"
    reset!(env)
    _sample(env, nothing, log_dir, file_name)
    data = load(joinpath(log_dir, file_name))
    for name in ["state", "input"]
        p = plot(data["time"], data[name])
        savefig(p, joinpath(log_dir, "custom_"*name*".pdf"))
    end
end

function _sample(env, agent, log_dir, file_name)
    logger = Logger(log_dir=log_dir, file_name=file_name, max_len=1000)
    obs = observe_flat(env)
    i = 0
    @time while true
        if agent == nothing
            action = nothing
        end
        next_obs, reward, done, info = env.step(action=action)
        record(logger, info)
        obs = next_obs
        i += 1
        if done
            break
        end
    end
    close!(env)
    close!(logger)
end

function test_all()
    test_Fym()
    test_largescale_env()
    test_custom_fym()
end

function test_deprecated()
    test_custom_env()
end

test_all()
