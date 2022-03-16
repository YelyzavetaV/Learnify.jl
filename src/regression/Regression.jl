module Regression

export enet!, ElasticNet, Lasso
export Logistic

include("base.jl")
include("enet.jl")
include("logistic.jl")

end