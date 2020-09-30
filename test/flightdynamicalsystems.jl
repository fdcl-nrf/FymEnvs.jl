using FymEnvs

function test_f16linearlateral()
    f16 = F16LinearLateral()
    systems = Dict("f16" => system(f16))
    env = BaseEnv()
    systems!(env, systems)
    dyn!(env, dyn(f16))
end


test_f16linearlateral()
