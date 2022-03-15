using Statistics

include("src/Learnify.jl")

using .Learnify

x = [5.0 2.0 3.0; -0.1 -3.0 -1.0; -2.0 3.0 3.0]
y = [1.0, 3.0, -0.9]

f = [2.0, 2.0]
q = [1, 1]
pts = [1.0 0.0; 3.0 5.0; 7.0 8.0]

model = Lasso(x, y)
println("Model converged in $(model.Nᵢ) iteration(s).")
println("Predicted coefficients are $(model.θ).")

# println(model(pts))
