module ReactiveObjectsTests
using Test, ReactiveObjects, Random, TestModules
include("ReactiveObjectsTests.jl")
end

using TestModules
runtests!(ReactiveObjectsTests)
