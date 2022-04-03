using Test
using Learnify: Classification as c

@testset "distance" begin
  X = [1.0 3.0
       7.0 4.0
       3.0 1.0]
  data = [5.0 3.0
          1.0 2.0]
  y = [4.0 1.0
       √5  √40
       √8  √5]
  @test c.distance(X, data) ≈ y
end
