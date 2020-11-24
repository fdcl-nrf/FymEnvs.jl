"""
module FymLogging

Logging module of FymEnvs.
"""

module FymLogging

import FymEnvs: close!, record

# using HDF5
using JLD2
using Dates
using Debugger

export Logger, close!, record, load, set_info!


############ Logger ############
mutable struct Logger
    path
    mode
    max_len
    info
    buffer
    len
    Logger(args...; kwargs...) = init!(new(), args...; kwargs...)
end

function init!(logger::Logger;
                    # path=nothing, log_dir=nothing, file_name="data.h5",
                    path=nothing, log_dir=nothing, file_name="data.jld2",
                    max_len=10, mode="w")
    if path == nothing
        if log_dir == nothing
            log_dir = joinpath("log", Dates.format(now(), "Ymmd-HMS"))
        end
        mkpath(log_dir)
        path = joinpath(log_dir, file_name)
    end
    logger.path = path
    # h5open(logger.path, mode) do dummy  # create .h5
    jldopen(logger.path, mode) do jldfile  # create .jld2
        JLD2.Group(jldfile, "data")  # make group
        JLD2.Group(jldfile, "info")  # make group
    end
    logger.mode = mode
    logger.max_len = max_len
    logger.info = Dict()
    clear!(logger)
    return logger
end

function clear!(logger::Logger)
    logger.buffer = Dict()
    logger.len = 0
end

"Record a dictionary or a numeric data preserving the structure."
function record(logger::Logger, info::Dict)
    _rec_update!(logger.buffer, info)
    logger.len += 1
    if logger.len >= logger.max_len
        flush!(logger)
    end
end

function flush!(logger::Logger; info=Dict())
    # h5open(logger.path, "r+") do h5file
    jldopen(logger.path, "r+") do h5file
        _rec_save!(h5file, "/data/", logger.buffer)
        # @enter _rec_save!(h5file, "/data/", logger.buffer)
        _info_save!(h5file, info)
    end
    clear!(logger)
end

"Close `logger`.
You must close manually defined logger after simulation terminated."
function close!(logger::Logger)
    flush!(logger; info=logger.info)
end

function set_info!(logger::Logger, info::Dict)
    _rec_update!(logger.info, info, is_info=true)
end


############ others ############
function _info_save!(h5file, info::Dict=Dict())
    for (key, val) in info
        h5file["info"][key] = val
        # attrs(h5file)[key] = val
    end
end

"Recursively save the `dic` into the HDF5 file."
function _rec_save!(h5file, path, dic)
    for (key, val) in dic
        if typeof(val) <: Array
            # if exists(h5file, path*key)
            val_tmp = h5file[path*key]
            val_new = cat(val, val_tmp, dims=1)
            if haskey(h5file, path*key)
                @bp
                h5file[path*key] = cat(h5file[path*key], val, dims=1)
                # dset = h5file[path*key]
                # dims = size(dset)
                # set_dims!(dset, (dims[1]+size(val)[1], dims[2:end]...))
                # indices_exceptone = [Colon()
                #                      for _ in 2:length(size(val))]
                # dset[end-size(val)[1]+1:end, indices_exceptone...] = val
            else
                # dset = JLD2.Group(h5file, path * key)
                # dset = d_create(h5file, path * key, Float64,
                #              (size(val), (-1, size(val)[2:end]...)),
                #              "chunk", size(val)
                #             )
                # indices_all = [Colon() for _ in 1:length(size(val))]
                # dset[indices_all...] = val
                indices_all = [Colon() for _ in 1:length(size(val))]
                h5file[path * key] = val
            end
        elseif typeof(val) <: Dict
            _rec_save!(h5file, path * key * "/", val)
        else
            error("Cannot save $(typeof(val)) type")
        end
    end
end

"Load HDF5 data saved by `Logger`."
function load(path; with_info=false)
    # h5open(path, "r") do h5file
    data = Dict()
    info = Dict()
    jldopen(path, "r") do file
        for key in keys(file["data"])
            data[key] = file["data"][key]
        end
        for key in keys(file["info"])
            info[key] = file["info"][key]
        end
    end

    if with_info
        return data, info
    else
        return data
    end
    # jldopen(path, "r") do h5file
    #     data = _rec_load(h5file, "data")
    #     # data = _rec_load(h5file, "/")
    #     if with_info
    #         # info = Dict()
    #         # attr = attrs(h5file)
    #         # for name in names(attr)
    #         #     info[name] = read(attr[name])
    #         # end
    #         info = h5file["info"]
    #         return data, info
    #     else
    #         return data
    #     end
    # end
end

function _rec_load(h5file, path)
    @bp
    data = read(h5file, path)
    return data
end

"Recursively update `base_dict` with `input_dict`."
function _rec_update!(base_dict, input_dict; is_info=false)
    for (key, val) in input_dict
        if typeof(val) <: Dict
            _rec_update!(get!(base_dict, key, Dict()), val,
                         is_info=is_info)
        else
            if is_info
                get!(base_dict, key, val)
            else
                if !(typeof(val) <: String)
                    get!(base_dict, key, [])
                    if length(size(val)) == 0
                        val = [val]
                    end
                    base_dict[key] = cat(base_dict[key],
                                         reshape(val, 1, size(val)...),
                                         dims=1)
                    # append!(get!(base_dict, key, []), val)
                else
                    error("Unsupported data type")
                end
            end
        end
    end
end

function raise_unsupported_error()
    error("Unsupported function yet")
end


end
