using FymEnvs
using Plots
ENV["GKSwstype"]="nul"  # do not show plot


function test_f16linearlateral()
    f16 = F16LinearLateral()

    function step(env; action=nothing)
        t = time(env.clock)
        sys = env.systems["f16"]
        x = sys.state
        u = zeros(2)
        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        info = Dict("time" => t, "state" => x, "input" => u)
        return next_obs, reward, done, info
    end
    function set_dyn(env, t; deriv=f16.deriv)
        sys = env.systems["f16"]
        x = sys.state
        u = zeros(2)
        sys.dot = deriv(x, u)
    end

    systems = Dict("f16" => f16.system)
    env = BaseEnv()
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    reset!(env)

    log_dir = "data/test"
    file_name = "f16.h5"
    _sample(env, nothing, log_dir, file_name)

    data = load(joinpath(log_dir, file_name))
    @show env
    @show size(data["state"])
    t, x = data["time"], data["state"]
    p = plot(t, x)
    savefig(p, joinpath(log_dir, "f16.pdf"))
end

function test_glidingvehicle3dof()
    gv3 = GlidingVehicle3DOF()

    function step(env; action=nothing)
        t = time(env.clock)
        sys = env.systems["gv3"]
        state = sys.state
        input = zeros(2)
        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        info = Dict("time" => t, "state" => state, "input" => input)
        return next_obs, reward, done, info
    end
    function set_dyn(env, t; deriv=gv3.deriv)
        sys = env.systems["gv3"]
        state = sys.state
        input = zeros(2)
        sys.dot = deriv(state, input)
    end

    systems = Dict("gv3" => gv3.system)
    env = BaseEnv()
    systems!(env, systems)
    dyn!(env, set_dyn)
    step!(env, step)
    reset!(env)

    log_dir = "data/test"
    file_name = "gv3.h5"
    _sample(env, nothing, log_dir, file_name)

    data = load(joinpath(log_dir, file_name))
    @show env
    @show size(data["state"]), size(data["input"])
    t, x = data["time"], data["state"]
    p = plot(t, x)
    savefig(p, joinpath(log_dir, "gv3.pdf"))
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


test_f16linearlateral()
test_glidingvehicle3dof()
