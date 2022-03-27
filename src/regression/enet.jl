using Statistics: mean
using LinearAlgebra: norm

"""
    enet!(θ::Vector{F}, X::Matrix{F}, y::Vector{F}, α, β;
        max_it::Integer=1000, ϵ=1e-5)

Elastic Net regression method.

# References
[1] S.-J. Kim et al. "An Interior-Point Method for Large-Scale l₁-Regularized Least
    Squares". IEEE J. Sel. Top. Signal Process. 1(4):606-617, 2007.
[2] F. Pedregosa et. al. "Scikit-learn: Machine Learning in Python". J. Mach. Learn. Res.
    12:2825-2830, 2011.
"""
function enet!(θ::Vector{F}, X::Matrix{F}, y::Vector{F}, α, β;
    max_it::Integer=1000, ϵ=1e-5) where {F}

    if (α == 0 && β == 0)
        @warn """It is not adviced to use Elastic Net algorithm with no regularization
                (α=0 and β=0)."""
    end
    if ~(0 ≤ β ≤ 1)
        throw(DomainError(β, "β must lie in range [0, 1]."))
    end

    M, N = size(X, 1), size(X, 2)
    λ = α * β * M
    γ = α * (1.0 - β) * M

    R = y - X * θ
    z = sum(X .^ 2, dims=1)

    ϵ̃ = ϵ * y' * y
    η = ϵ̃ + 1.0

    it = 0
    converged = false
    while (~converged && it < max_it)
        it += 1
        θₘ, 𝝙 = 0.0, 0.0
        for j = 1:N
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
        if (θₘ == 0 || 𝝙 / θₘ < ϵ || it == max_it)
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
            maximal allowed iterations max_it or decreasing the tolerance ϵ."""
    end
    return it, η
end

function preprocess!(X::Matrix{F}, y::Vector{F}, intercept::Bool=true) where {F}
    X̄ = zeros(F, size(X, 2))
    ȳ = 0.0
    if intercept
        X̄ = mean(X, dims=1)
        ȳ = mean(y)
        X .-= X̄
        y .-= ȳ
    end
    return X̄, ȳ
end

"""
    ElasticNet(
        X::Matrix{F},
        y::Vector{F};
        α=1.0,
        β=0.5,
        intercept::Bool=true,
        max_it::Integer=1000,
        ϵ=1e-5,
    )

Elastic Net regression model that solves the optimization problem
    min[R(θ)] = min[1/(2M)‖y - Xθ‖² + αβ‖θ‖₁ + (1/2)α(1 - β)‖θ‖²],
where X and y are the training data with M samples and N features, and ‖⋅‖₁ and ‖⋅‖
denote l₁ and l₂ norms, respectively.
If `intercept` is true, training data will be centered.
`max_it` is a maximal number of iterations and `ϵ` is a convergence tolerance for the
duality gap.

# References
[1] J. Friedman et al. "Regularization Paths for Generalized Linear Models via Coordinate
    Descent". J. Stat. Softw. 33(1):1-22, 2010.
"""
mutable struct ElasticNet{F} <: AbstractLinearRegression
    I::F
    θ::Vector{F}
    nit::Integer
    η
    function ElasticNet(
        X::Matrix{F},
        y::Vector{F};
        α=1.0,
        β=0.5,
        intercept::Bool=true,
        max_it::Integer=1000,
        ϵ=1e-5,
    ) where {F}
        X̄, ȳ = preprocess!(X, y, intercept)
        θ = zeros(F, size(X, 2))

        nit, η = enet!(θ, X, y, α, β; max_it, ϵ)

        I = (ȳ .- X̄ * θ)[1]

        new{F}(I, θ, nit, η)
    end
end

function (m::ElasticNet)(pts::AbstractVecOrMat)
    pts * m.θ .+ m.I
end

"""
    Lasso(
        X::Matrix{F},
        y::Vector{F};
        α=1.0,
        intercept::Bool=true,
        max_it::Integer=1000,
        ϵ=1e-5,
    )
"""
function Lasso(
    X::Matrix{F},
    y::Vector{F};
    α=1.0,
    intercept::Bool=true,
    max_it::Integer=1000,
    ϵ=1e-5,
) where {F}
    ElasticNet(X, y; α=α, β=1.0, intercept=intercept, max_it=max_it, ϵ=ϵ)
end
