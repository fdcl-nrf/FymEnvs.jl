using FymEnvs
using Plots
ENV["GKSwstype"]="nul"  # do not show plot


function test_f16linearlateral()
    function step!(env)
        t = time(env.clock)
        sys = env.systems["f16"]
        x = sys.state
        update!(env)
        next_obs = sys.state
        reward = zeros(1)
        done = time_over(env.clock)
        info = Dict()
        return next_obs, reward, done, info
    end
    f16 = F16LinearLateral()
    function set_dyn(env, t; deriv=dyn(f16))
        sys = env.systems["f16"]
        x = sys.state
        u = zeros(2)
        sys.dot = deriv(x, u)
    end
    systems = Dict("f16" => system(f16))
    log_dir = "data"
    file_name = "test.h5"
    logger = Logger(log_dir=log_dir, file_name=file_name)
    env = BaseEnv(logger=logger, max_t=0.10)
    systems!(env, systems)
    dyn!(env, set_dyn)
    reset!(env)
    obs = observe_flat(env)
    i = 0
    @time while true
        next_obs, reward, done, info = step!(env)
        obs = next_obs
        i += 1
        if done
            break
        end
    end
    close!(env)
    data = load(env.logger.path)
    @show env
    @show size(data["state"]["f16"])
    t, x = data["time"], data["state"]["f16"]
    p = plot(t, x)
    savefig(p, "data/f16.pdf")
end


test_f16linearlateral()
