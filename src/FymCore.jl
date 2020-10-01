"""
module FymCore

The main core codes for `FymEnvs`.

"""
module FymCore

using Reexport
using Revise


import FymEnvs: close!

include("FymLogging.jl")
@reexport using .FymLogging

using ProgressMeter
# using Debugger  # TODO: delete it

using StaticArrays

# export FymEnv
export BaseEnv
# export FymSystem
export BaseSystem
export Clock

export update!, close!, reset!, render
export observe_array, observe_dict, observe_flat
export systems!, dyn!, step!
export time_over, time


############ Clock ############
mutable struct Clock
    t::Float64
    dt::Float64
    max_t::Float64
    max_len::Int64
    thist::Array
    ode_step_len::Int64
    Clock(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(clock::Clock, dt, ode_step_len; max_t=10.0)
    clock.dt = Float64(dt)
    clock.max_t = Float64(max_t)
    clock.max_len = floor(Int, (max_t / dt) + 1)
    clock.ode_step_len = Int64(ode_step_len)
    clock.thist = collect(range(0.0,
                                stop=dt, length=clock.ode_step_len + 1))
    return clock
end

function reset!(clock::Clock)
    clock.t = 0.0
    return clock
end

function tick!(clock::Clock)
    clock.t += clock.dt
end

function time!(clock::Clock, t)
    clock.t = t  # array
end

function Base.time(clock::Clock)
    return clock.t
end

function time_over(clock::Clock; t=nothing)
    if t == nothing
        return time(clock) >= clock.max_t
    else
        return t >= clock.max_t
    end
end

function thist(clock::Clock)
    thist = clock.thist .+ time(clock)
    if time_over(clock, t=thist[end])
        index = findfirst(thist .> clock.max_t)
        if index == nothing
            return thist
        else
            return thist[1:index[1]]
        end
    else
        return thist
    end
end

############ BaseSystem ############
mutable struct BaseSystem
    initial_state::Array
    state::Array
    state_size::Tuple
    state_length::Int64
    flat_index::AbstractRange
    dot::Array
    name::Union{String, Nothing}
    BaseSystem(; kwargs...) = init!(new(); kwargs...)
end

function _show(sys::BaseSystem; i=0)
    result = []
    for symbol in [:name, :state, :dot, :initial_state,
                   :state_size, :flat_index]
        if isdefined(sys, symbol)
            value = getproperty(sys, symbol)
        else
            value = "undef"
        end
        if symbol == :name
            space = "+---"
        else
            space = "|   "
        end
        push!(result,
              _add_space("$(String(symbol)): $value", i, space=space))
    end
    return join(result, "\n")
end

function Base.show(io::IO, sys::BaseSystem)
    res = _show(sys)
    println(io, res)
end

function init!(sys::BaseSystem; initial_state=nothing,
               state_size=(1, 1), name=nothing)
    if initial_state == nothing
        initial_state = zeros(state_size)
    end
    sys.initial_state = convert(Array{Float64}, initial_state)
    sys.state_size = size(sys.initial_state)
    sys.name = name
    return sys
end

function reset!(sys::BaseSystem)
    state!(sys, sys.initial_state)
    return sys.state
end

function state(sys::BaseSystem)
    return sys.state
end

function state!(sys::BaseSystem, state)
    sys.state = state
end

function dot(sys::BaseSystem)
    return sys.dot
end

function dot!(sys::BaseSystem, dot)
    sys.dot = dot
end


############ BaseEnv ############
mutable struct BaseEnv
    name
    systems::Dict{Any, Union{BaseEnv, BaseSystem}}
    dyn
    step

    dt::Float64
    clock::Clock
    progressbar
    logger

    solver
    ode_func
    ode_option::Dict
    dot::Array
    flat_index::AbstractRange
    state_length::Int
    state_size::Tuple

    params::Dict

    BaseEnv(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function _add_space(string, i; space=" "^4)
    return space^i * string
end

function _show(env::BaseEnv; i=0)
    result = []
    push!(result, _add_space("name: $(env.name)", i, space="+---"))
    for system in _systems(env)
        if typeof(system) == BaseSystem
            v_str = _show(system, i=i+1)
        elseif typeof(system) == BaseEnv
            v_str = _show(system, i=i+1)
        end
        push!(result, v_str)
    end
    res = join(result, "\n")
end

function Base.show(io::IO, env::BaseEnv)
    res = _show(env)
    println(res)
end

function init!(env::BaseEnv;
               systems=Dict(), dyn=nothing, step=nothing, params=Dict(),
               dt=0.01, max_t=1.0, ode_step_len=1,
               logger=nothing, ode_option=Dict(), solver="rk4",
               name=nothing,
              )
    env.name = name
    systems!(env, systems)

    env.clock = Clock(dt, ode_step_len, max_t=max_t)
    env.dt = env.clock.dt

    env.logger = logger

    if solver == "rk4"
        env.solver = rk4
    else
        error("Unsupported solver")
    end
    dyn!(env, dyn)
    step!(env, step)
    env.ode_func = ode_wrapper(env)
    env.ode_option = ode_option
    env.progressbar = nothing
    env.params = params
    return env
end

function ode_wrapper(env::BaseEnv)
    wrapper = function(y, kwargs, t)
        for system in _systems(env)
            state!(system, reshape(y[system.flat_index],
                                   system.state_size))
        end
        env.dyn(env, t; kwargs...)
        res = vcat([dot(system)[:] for system in _systems(env)]...)
        return SVector{env.state_size[1]}(res)
        # return res
    end
    return wrapper
end

function indexing!(env::BaseEnv)
    start = 0
    for system in _systems(env)
        system.state_length = prod(system.state_size)
        system.flat_index = (start+1):(start + system.state_length)
        start += system.state_length
    end
    tmp = [system.state_length for system in _systems(env)]
    if length(tmp) == 0
        env.state_size = (0,)
    else
        env.state_size = (sum(tmp),)
    end
end

function reset!(env::BaseEnv)
    for system in _systems(env)
        reset!(system)
    end
    reset!(env.clock)
end

function state(env::BaseEnv)
    return observe_flat(env)
end

function _systems(env::BaseEnv)
    return values(systems(env))
end

function systems(env::BaseEnv)
    return env.systems
end

function systems!(env::BaseEnv, systems::Dict)
    env.systems = systems
    indexing!(env)
end

function dyn!(env::BaseEnv, dyn)
    env.dyn = dyn
end

function step!(env::BaseEnv, step)
    _step(args...; kwargs...) = step(env, args...; kwargs...)
    env.step = _step
end

function dyn(env::BaseEnv)
    return env.dyn
end

function state!(env::BaseEnv, state)
    for system in _systems(env)
        state!(system, reshape(state[system.flat_index],
                               system.state_size))
    end
end

function dot(env::BaseEnv)
    return vcat([dot(system)[:]
                 for system in _systems(env)]...)  # flatten
end

function dot!(env::BaseEnv, dot)
    for system in _systems(env)
        system.dot = reshape(dot[system.flat_index], system.state_size)
    end
end

function observe_array(env::BaseEnv; state=nothing)
    res = []
    if state == nothing
        for system in _systems(env)
            if typeof(system) == BaseSystem
                push!(res, state(system))
            elseif typeof(system) == BaseEnv
                push!(res, observe_array(system))
            end
        end
    else
        raise_unsupported_error()
    end
    return res
end

function observe_dict(env::BaseEnv; state=nothing)
    res = Dict()
    if state == nothing
        for (name, system) in systems(env)
            if typeof(system) == BaseSystem
                res[name] = state(system)
            elseif typeof(system) == BaseEnv
                res[name] = observe_dict(system)
            end
        end
    else
        for (name, system) in systems(env)
            if typeof(system) == BaseSystem
                res[name] = reshape(state[system.flat_index],
                                    system.state_size)
            elseif typeof(system) == BaseEnv
                res[name] = observe_dict(system,
                                         state=state[system.flat_index])
            end
        end
    end
    return res
end

function observe_flat(env::BaseEnv)
    return vcat([state(system)[:]
                 for system in _systems(env)]...)  # flatten
end

function ProgressMeter.update!(env::BaseEnv; kwargs...)
    t_hist = thist(env.clock)
    ode_hist = env.solver(
                          env.ode_func,
                          observe_flat(env),
                          t_hist,
                          kwargs,
                          env.ode_option...,
                         )
    done = false
    # TODO: low priority; add eager stop; should we?
    tfinal, yfinal = t_hist[end], ode_hist[end]
    # Update the systems' state
    for system in _systems(env)
        state!(system, reshape(yfinal[system.flat_index],
                               system.state_size))
    end

    # TODO: low priority; Log the inner history of states
    if env.logger != nothing
        for (t, y) in zip(t_hist[1:end-1], ode_hist[1:end-1])
            state_dict = observe_dict(env, state=y)
            _info = Dict("time" => t, "state" => state_dict)
            if kwargs != nothing
                if haskey(Dict(kwargs), "time") || haskey(Dict(kwargs),
                                                          "state")
                    error("invalid kwargs")
                else
                    info = merge(_info, Dict(kwargs))
                end
            else
                info = _info
            end
            record(env.logger, info)
        end
        # TODO: low priority; add logger_callback
    end
    time!(env.clock, tfinal)

    # TODO: low priority; add delay; should we?
    return t_hist, ode_hist, done || time_over(env.clock)
end

function close!(env::BaseEnv)
    if env.logger != nothing
        close!(env.logger)
    end
end

function render(env::BaseEnv;
                mode="ProgressMeter", desc="", dt=0.01, kwargs...)
    if mode == "ProgressMeter"
        if env.progressbar == nothing || time(env.clock) == 0
            env.progressbar = Progress(env.clock.max_len, dt,
                                       desc, kwargs...)
        end
        next!(env.progressbar)
    end
end

function set_delay(env::BaseEnv, systems::Array, T)
    raise_unsupported_error()
end

function raise_unsupported_error()
    error("Unsupported function yet")
    # TODO
end

function rk4(func, y0, t, kwargs)
    n = length(t)
    y = [zero(y0) for i in 1:n]
    y[1] = y0
    for i in 1:n-1
        h = t[i+1] - t[i]
        k1 = func(y[i], kwargs, t[i])
        k2 = func(y[i] + k1 * h / 2.0, kwargs, t[i] + h / 2.0)
        k3 = func(y[i] + k2 * h / 2.0, kwargs,  t[i] + h / 2.0)
        k4 = func(y[i] + k3 * h, kwargs, t[i] + h)
        y[i+1] = y[i] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    end
    return y
end


end
