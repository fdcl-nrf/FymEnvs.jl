# NOTICE
This repository will be deprecated.
For the next version, see [LazyFym](https://github.com/JinraeKim/LazyFym.jl).

# FymEnvs
**FymEnvs.jl** is a Julia version of [the original `fym`](https://github.com/fdcl-nrf/fym),
developed by FDCL in SNU.

You can perform numerical (flight) simulations including deep reinforcement learning with this package.
This package is also highly inspired by [`Gym`](https://gym.openai.com/), OpenAI.

## Usage
The usage of **FymEnvs.jl** is very similar to `fym`,
but there is a significant difference: *does not inherit BaseEnv* like class in `Python`.
It may be awkward to the users from the original `fym`.
**NOTICE**: To create your own `FymEnv`,
it is *highly recommended* to see
the contents about custom environments in directory `test`.

Here is the simplest example of **FymEnvs.jl**:

```julia
using FymEnvs
using LinearAlgebra

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
env = BaseEnv(max_t=100.00, logger=logger, name="test_env",)
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
```

Result:

```julia
# time and progressbar
 99%|████████████████████████████████████████████████████████████████████████████████████▎|  ETA: 0:00:00  0.706294 seconds (2.98 M allocations: 1.675 GiB, 11.48% gc time)
# representation, i.e., show (nested env supported)
name: test_env
max_t: 100.0
dt: 0.01
+---name: 3d_sys
|   state: [3.7200760072278747e-44, 7.440152014455749e-44, 1.1160228021683672e-43]
|   dot: [-3.7200756925403154e-44, -7.440151385080631e-44, -1.1160227077620995e-43]
|   initial_state: [1.0, 2.0, 3.0]
|   state_size: (3,)
|   flat_index: 1:3
# saved data
(10000, 3)
```

For more examples, see directory `test`.

## Notice
### **FymEnvs.jl** does not directly support `Gym`'s features.
- Note that this does not inherit `Gym`'s features, while `fym` *does*.
### Supported features
The following features are supported in **FymEnvs.jl**:
- Nested environments
    - An environment can contain other environments as its systems.
- Backward integration
    - Can perform backward integration with keyword argument `dt (< 0.0).`
- Log data
    - Can log simulation data and configuration using `Logger`.
    - It is compatible with [JLD2.jl](https://github.com/JuliaIO/JLD2.jl).

### Not supported features
There are some features of `fym`, not realised yet. Here's the list:
- eager stop
- logger callback
- delayed system
