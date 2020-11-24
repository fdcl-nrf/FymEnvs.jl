using FymEnvs
using LinearAlgebra

if !isdefined(Main, :TestEnvs)
    # using Revise; includet("custom_env.jl")  # to avoid conflict
    include("custom_env.jl")  # to avoid conflict
    using .TestEnvs
end


function print_msg(test_name)
    println(">"^6*" "*test_name*" "*"<"^6)
end


function test_custom_fym()
    fym = TestEnv()

    env = fym.env
    log_dir = "data/test"
    file_name = "custom.jld2"
    # file_name = "custom.h5"
    reset!(env)
    _sample(fym, nothing, log_dir, file_name)
    data, info = load(joinpath(log_dir, file_name),
                      with_info=true)
    @bp
    _sample2(fym, nothing, log_dir, file_name)
    data, info = load(joinpath(log_dir, file_name),
                      with_info=true)
    @bp
end

function _sample(fym::FymEnv, agent, log_dir, file_name)
    env = fym.env
    logger = Logger(log_dir=log_dir, file_name=file_name, max_len=10)
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
    logger = Logger(log_dir=log_dir, file_name=file_name, max_len=1000)
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

# @enter test_custom_fym()
test_custom_fym()
