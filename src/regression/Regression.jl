module Regression

include("linear.jl")
include("enet.jl")

using .Linear: ElasticNet
using .Enet: enet!

export enet!, ElasticNet

end