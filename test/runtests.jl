using FymEnvs
using LinearAlgebra

using Plots
ENV["GKSwstype"]="nul"  # do not show plot
# using Debugger

if !isdefined(Main, :TestEnvs)
    # using Revise; includet("custom_env.jl")  # to avoid conflict
    include("custom_env.jl")  # to avoid conflict
    using .TestEnvs
end


function print_msg(test_name)
    println(">"^6*" "*test_name*" "*"<"^6)
end

function test_custom_fym()
    print_msg("custom FymEnv")
    fym = TestEnv(; dt=0.01, max_t=0.1)

    env = fym.env
    log_dir = "data/test"
    file_name = "custom.h5"
    reset!(env)
    _sample(fym, nothing, log_dir, file_name)
    data, info = load(joinpath(log_dir, file_name),
                      with_info=true)
    @show typeof(data["done"])
    plot(data["time"], data["state"])
    for name in ["state", "input"]
        p = plot(data["time"], data[name])
        savefig(p, joinpath(log_dir, "custom_"*name*".pdf"))
    end
end

function test_reverse_time()
    print_msg("reverse_time")
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
    env = BaseEnv(max_t=0.00, dt=-0.01, initial_time=1.0, logger=logger, name="test_env",)
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

    show(env)
    show(size(data["state"]["sys"]))
end

function _sample(env::BaseEnv, agent, log_dir, file_name)
    logger = Logger(log_dir=log_dir, file_name=file_name)
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

function _sample(fym::FymEnv, agent, log_dir, file_name)
    env = fym.env
    logger = Logger(log_dir=log_dir, file_name=file_name)
    obs = observe_flat(env)
    i = 0
    @time while true
        if agent == nothing
            action = nothing
            if time(env.clock) > 0.5
                fym.controller.K = - 3*Matrix(I, 3, 3)
            end
        end
        next_obs, reward, done, info = env.step(action=action)
        record(logger, info)
        obs = next_obs
        i += 1
        if done
            config = Dict(
                          "name" => env.name,
                       )
            set_info!(logger, config)
            break
        end
    end
    close!(env)
    close!(logger)
end

function test_all()
    test_custom_fym()
    test_reverse_time()
end

test_all()
