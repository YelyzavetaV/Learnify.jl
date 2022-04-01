export distance

function distance(x::Matrix{F}, pts...) where {F}
    dists = Vector{F}[]

    for i = 1:size(y, 1)
        z = x .- (yt[:, i])'
    end
    return z
end