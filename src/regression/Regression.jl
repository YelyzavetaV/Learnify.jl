module Regression

include("linear.jl")
include("enet.jl")

using .Linear: ElasticNet, Lasso
using .Enet: enet!

export enet!, ElasticNet, Lasso

end