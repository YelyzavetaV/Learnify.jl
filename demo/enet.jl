include("../src/Learnify.jl")

using .Learnify

x = [5.0 2.0 3.0; -0.1 -3.0 -1.0; -2.0 3.0 3.0]
y = [1.0, 3.0, -0.9]

pts = [1.0 0.0; 3.0 5.0; 7.0 8.0]

model = ElasticNet(x[:, :], y; α=0)
print("Model converged in $(model.nit) iteration(s).\n")
print("Predicted coefficients are $(model.θ).")

# println(model(pts))