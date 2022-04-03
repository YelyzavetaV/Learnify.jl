@enum DataStructure begin
    naive
    kdtree
end

mutable struct NearestNeighbours{F} <: AbstractClassification
    X::Matrix{F}
    y::Vector{F}
    k::Integer
    ds::DataStructure
    function NearestNeighbours(
        X::Matrix{F},
        y::Vector{F};
        k::Integer=10,
        ds::DataStructure=naive,
    ) where {F}
        new{F}(X, y, k, ds)
    end
end

function (m::NearestNeighbours)(pts::AbstractVecOrMat)
end
