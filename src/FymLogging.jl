"""
module FymLogging

Logging module of FymEnvs.
"""

module FymLogging

import FymEnvs: close!, record

using JLD2
using Dates
# using Debugger

export Logger, close!, record, load, set_info!


############ Logger ############
mutable struct Logger
    path
    mode
    info
    buffer
    len
    max_len
    Logger(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(logger::Logger;
               path=nothing, log_dir=nothing, file_name="data.jld2",
               max_len::Int=1000,
               mode="w")
    if path == nothing
        if log_dir == nothing
            log_dir = joinpath("log", Dates.format(now(), "Ymmd-HMS"))
        end
        mkpath(log_dir)
        path = joinpath(log_dir, file_name)
    end
    logger.path = path
    logger.max_len = max_len
    jldopen(path, mode) do jldfile
        initialise(jldfile, ["data", "info"])
    end
    logger.mode = mode
    logger.info = Dict()
    clear!(logger)
    return logger
end

function initialise(jldfile, group_array)
    JLD2.Group(jldfile, "data")
    JLD2.Group(jldfile, "info")
end

function clear!(logger::Logger)
    logger.buffer = Dict()
    logger.len = 0
end

"Record a dictionary or a numeric data preserving the structure."
function record(logger::Logger, info::Dict)
    _rec_update!(logger.buffer, info)
    logger.len += 1
    if logger.len % logger.max_len == 0
        flush!(logger)
    end
end

function append_data(data, buffer)
    for key in keys(buffer)
        if key in keys(data)
            if typeof(data[key]) <: Dict
                _ = append_data(data[key], buffer[key])
            else
                data[key] = cat(data[key], buffer[key], dims=1)
            end
        else
            data[key] = buffer[key]
        end
    end
    return data
end

function flush!(logger::Logger)
    data = load(logger.path; with_info=false)
    # data appending
    append_data(data, logger.buffer)
    # info update
    info = logger.info  # always overwrited
    jldopen(logger.path, "w") do jldfile
        initialise(jldfile, ["data", "info"])
        _rec_save!(jldfile, "/data/", data)
        _info_save!(jldfile, info)
    end
    clear!(logger)
end

"Close `logger`.
You must close manually defined logger after simulation terminated."
function close!(logger::Logger)
    flush!(logger)
end

function set_info!(logger::Logger, info::Dict)
    _rec_update!(logger.info, info, is_info=true)
end


############ others ############
function _info_save!(jldfile, info::Dict=Dict())
    for (key, val) in info
        jldfile["info"][key] = val
    end
end

"Recursively save the `dic` into the JLD2 file."
function _rec_save!(jldfile, path, dic)
    for (key, val) in dic
        if typeof(val) <: Array
            jldfile[path * key] = val
        elseif typeof(val) <: Dict
            _rec_save!(jldfile, path * key * "/", val)
        else
            error("Cannot save $(typeof(val)) type")
        end
    end
end

"Load JLD2 data saved by `Logger`."
function load(path; with_info=false)
    data = Dict()
    info = Dict()
    jldopen(path, "r") do jldfile
        data = _rec_load(jldfile, "data")
        if with_info
            info = _rec_load(jldfile, "info")
        end
    end
    if with_info
        return data, info
    else
        return data
    end
end

function _rec_load(jldfile, path)
    ans = Dict()
    for key in keys(jldfile[path])
        val = jldfile[path * "/" * key]
        if typeof(val) <: JLD2.Group
            ans[key] = _rec_load(jldfile, path * "/" * key)
        else
            ans[key] = val
        end
    end
    return ans
end

"Recursively update `base_dict` with `input_dict`."
function _rec_update!(base_dict, input_dict; is_info=false)
    for (key, val) in input_dict
        if typeof(val) <: Dict
            _rec_update!(get!(base_dict, key, Dict()), val; is_info=is_info)
        else
            if is_info
                get!(base_dict, key, val)
            else
                if !(typeof(val) <: String)
                    get!(base_dict, key, typeof(val[1])[])
                    if length(size(val)) > 0
                        val = reshape(val, 1, size(val)...)
                    end
                    base_dict[key] = cat(base_dict[key], val, dims=1)
                else
                    error("Unsupported data type")
                end
            end
        end
    end
end


end
