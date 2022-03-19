using Test
using Learnify: ElasticNet
using Learnify: Regression

X = ([0.0; 0.0; 0.0])[:, :]
y = [0.0, 0.0, 0.0]
@testset "enet_throws" begin
    @test_throws DomainError ElasticNet(X, y; β=-1)
end

@testset "preprocess_no_intercept" begin
    X̄, ȳ = Regression.preprocess!(X, y, false)
    @test (X̄ == [0.0] && ȳ == 0.0)
end
