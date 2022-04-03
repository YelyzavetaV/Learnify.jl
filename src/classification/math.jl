using LinearAlgebra

export distance

function distance(x::Matrix{F}, data::VecOrMat{F}) where {F}
    dists = Matrix{F}(undef, size(x, 1), size(data, 1))
    for (i, d) in enumerate(eachrow(data))
        dists[:, i] = mapslices(norm, x .- d', dims=2)
    end
    return dists
end
