module Enet

using LinearAlgebra: norm

"""
    enet!(θ::Vector{F}, X::Matrix{F}, y::Vector{F}, α, β;
        iₘ::Integer=1000, ϵ=1e-5)

Elastic Net regression method.

# References
[1] S.-J. Kim et al. "An Interior-Point Method for Large-Scale l₁-Regularized Least
    Squares". IEEE J. Sel. Top. Signal Process. 1(4):606-617, 2007.
[2] F. Pedregosa et. al. "Scikit-learn: Machine Learning in Python". J. Mach. Learn. Res.
    12:2825-2830, 2011.
"""
function enet!(θ::Vector{F}, X::Matrix{F}, y::Vector{F}, α, β;
    itₘ::Integer=1000, ϵ=1e-5) where {F}

    N, M = size(X, 1), size(X, 2)
    λ = α * β * N
    γ = α * (1.0 - β) * N

    R = y - X * θ
    z = sum(X .^ 2, dims=1)

    ϵ̃ = ϵ * y' * y
    η = ϵ̃ + 1.0

    it = 0
    converged = false
    while (~converged && it < itₘ)
        it += 1
        θₘ, 𝝙 = 0.0, 0.0
        for j = 1:M
            z[j] == 0 && continue
            θ̃ = θ[j]
            if θ̃ ≠ 0
                R += X[:, j] * θ̃
            end
            G = X[:, j]' * R
            θ[j] = sign(G) * max(abs(G) - λ, 0) / (z[j] + γ)
            if θ[j] ≠ 0
                R -= X[:, j] * θ[j]
            end
            𝝙 = max(𝝙, abs(θ[j] - θ̃))
            θₘ = max(θₘ, abs(θ[j]))
        end
        if (θₘ == 0 || 𝝙 / θₘ < ϵ || it == itₘ)
            c₁ = c₂ = c₃ = 1
            d = maximum(abs.(X' * R - γ * θ))
            if d > λ
                s = λ / d
                c₁ -= 0.5 * (1.0 - s^2)
                c₂ = c₃ = s
            end
            η = (
                c₁ * R' * R
                - c₂ * R' * y
                + λ * norm(θ, 1)
                + 0.5 * γ * (1.0 + c₃^2) * (θ' * θ)
            )
            if η < ϵ̃
                converged = true
            end
        end
    end
    if ~converged
        @warn """Elastic Net algorithm did not converge: try increasing the number of
            maximal allowed iterations itₘ or decreasing the tolerance ϵ."""
    end
    return it, η
end

end