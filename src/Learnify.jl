module Learnify

include("regression/Regression.jl")
include("classification/Classification.jl")
include("generate/Generate.jl")
include("utility/Utility.jl")

using .Regression
export ElasticNet, Lasso
export Logistic

using .Classification
export euclidian
export NearestNeighbours

using .Generate
using .Utility

end