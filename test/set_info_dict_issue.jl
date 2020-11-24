using FymEnvs
using LinearAlgebra

if !isdefined(Main, :TestEnvs)
    include("custom_env.jl")  # to avoid conflict
    using .TestEnvs
end


function test_custom_fym()
    fym = TestEnv(; max_t=0.1)

    env = fym.env
    log_dir = "data/test"
    file_name = "custom.jld2"
    reset!(env)
    _sample(fym, nothing, log_dir, file_name)
    data, info = load(joinpath(log_dir, file_name),
                      with_info=true)
    print(info)

    reset!(env)
    _sample2(fym, nothing, log_dir, file_name)
    data, info = load(joinpath(log_dir, file_name),
                      with_info=true)
    print(info)
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

function _sample2(fym::FymEnv, agent, log_dir, file_name)
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
                          "name" => Dict(),
                       )
            set_info!(logger, config)
            break
        end
    end
    close!(env)
    close!(logger)
end

test_custom_fym()
