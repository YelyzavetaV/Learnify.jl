using LinearAlgebra
using BenchmarkTools
using Random
import Base.-

include("src/Learnify.jl")

using .Learnify
using .Learnify: Classification as cl

x = [1.0 3.0
     7.0 4.0
     3.0 1.0]
pts = [5.0 3.0
       1.0 2.0]
# x = randn(100, 100)
# pts = randn(50, 100)

dists = cl.distance(x, pts)
# rdists = cl.rdistance(x, pts)
# @btime rdists = cl.rdistance(x, pts)
# @btime dists = cl.distance(x, pts)

print(dists, "\n")
# print(rdists, "\n")

# model = NearestNeighbours(x, y)
# println(model.ds)
