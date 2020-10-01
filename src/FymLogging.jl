"""
module FymLogging

Logging module of FymEnvs.
"""

module FymLogging

import FymEnvs: close!, record

using HDF5
using Dates

export Logger, close!, record, load


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
                    path=nothing, log_dir=nothing, file_name="data.h5",
                    max_len=10, mode="w")
    if path == nothing
        if log_dir == nothing
            log_dir = joinpath("log", Dates.format(now(), "Ymmd-HMS"))
        end
        mkpath(log_dir)
        path = joinpath(log_dir, file_name)
    end
    logger.path = path
    h5open(logger.path, mode) do dummy  # create .h5
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

function record(logger::Logger, info::Dict)
    """Record a dictionary or a numeric data preserving the structure."""
    _rec_update!(logger.buffer, info)
    logger.len += 1
    if logger.len >= logger.max_len
        flush!(logger)
    end
end

function flush!(logger::Logger; info=nothing)
    h5open(logger.path, "r+") do h5file
        _rec_save!(h5file, "/", logger.buffer)
        # _info_save!(h5file, info)
    end
    clear!(logger)
end

function close!(logger::Logger)
    flush!(logger, info=logger.info)
end

function set_info!(logger::Logger, args...; kwargs...)
    _rec_update!(logger.info, Dict(zip(args, kwargs)))
    h5open(logger.path, "r+") do h5file
        # _info_save(h5file, logger.info)
    end
end


############ others ############
function _info_save!(h5file; info=nothing)
    # TODO: low priority
    raise_unsupported_error()
end

function _rec_save!(h5file, path, dic)
    """Recursively save the ``dic`` into the HDF5 file."""
    for (key, val) in dic
        if typeof(val) <: Array
            if exists(h5file, path*key)
                dset = h5file[path*key]
                dims = size(dset)
                set_dims!(dset, (dims[1]+size(val)[1], dims[2:end]...))
                indices_exceptone = [Colon() for _ in 2:length(size(val))]
                dset[end-size(val)[1]+1:end, indices_exceptone...] = val
            else
                dset = d_create(h5file, path * key, Float64,
                             (size(val), (-1, size(val)[2:end]...)),
                             "chunk", size(val)
                            )
                indices_all = [Colon() for _ in 1:length(size(val))]
                dset[indices_all...] = val
            end
        elseif typeof(val) <: Dict
            _rec_save!(h5file, path * key * "/", val)
        else
            error("Cannot save $(typeof(val)) type")
        end
    end
end

function load(path; with_info=false)
    h5open(path, "r") do h5file
        ans = _rec_load(h5file, "/")
        if with_info
            # TODO: low priority
            raise_unsupported_error()
        else
            return ans
        end
    end
end

function _rec_load(h5file, path)
    # TODO: check this
    ans = read(h5file[path])
    return ans
end

function _rec_update!(base_dict, input_dict; is_info=false)
    """Recursively update ``base_dict`` with ``input_dict``."""
    for (key, val) in input_dict
        if typeof(val) <: Dict
            _rec_update!(get!(base_dict, key, Dict()), val, is_info=is_info)
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
