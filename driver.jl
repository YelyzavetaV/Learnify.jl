using LinearAlgebra
using BenchmarkTools
using Random
import Base.-

include("src/Learnify.jl")

using .Learnify
using .Learnify: Utility as utils

x = [1.0 3.0; 6.0 10.0 ; 0.0 0.0; 9.0 -3.0]
pts = [0.0 0.0; 5.0 3.0]
# x = randn(100, 100)
# pts = randn(50, 100)
y = [0.0, 0.0, 1.0, 1.0]

print(y[[1, 3]])

# @btime utils.matnorm(x, pts)
# print(z ≈ z2)

# print(dist, "\n")
# print(norm([1.0, 3.0] - [5.0, 3.0]))

# model = NearestNeighbours(x, y)
# println(model.ds)
