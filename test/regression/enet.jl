using Test
using Learnify: ElasticNet

x = [5.0 2.0 3.0; -0.1 -3.0 -1.0; -2.0 3.0 3.0]
y = [1.0, 3.0, -0.9]
@testset "enet_throws" begin
    @test_throws DomainError ElasticNet(x, y; β=-1)
end
