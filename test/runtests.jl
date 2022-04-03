using Test

@testset "Regression" begin
    include("regression/enet.jl")
end

@testset "Classification" begin
    include("classification/math.jl")
end
