using Strada
using Compat
using Base.Test

blob = Strada.CaffeDict()
blob["conv1"] = Any[ones(3,3), ones(3)]
blob["pool1"] = Any[ones(5,5)]

@test length(blob) == 3*3 + 3 + 5*5

# Test roundtrip

theta = randn(length(blob))

copy!(blob, theta)

newtheta = zeros(length(blob))

copy!(newtheta, blob)

@test norm(newtheta - theta) <= 1e-8
