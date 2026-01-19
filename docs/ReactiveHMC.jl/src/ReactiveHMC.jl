module ReactiveHMC

using ReactiveObjects, LinearAlgebra, LazyArrays

export leapfrog!, generalized_leapfrog!, implicit_midpoint!, multistep
export euclidean_phasepoint, riemannian_phasepoint, riemannian_softabs_phasepoint, relativistic_euclidean_phasepoint, relativistic_riemannian_phasepoint, relativistic_riemannian_softabs_phasepoint
export @dottilde

macro dottilde(x)
    @assert Meta.isexpr(x, :(=))
    lhs, rhs = x.args
    esc(:($lhs .= @~ $rhs)) 
end

include("integrators.jl")
include("phasepoints.jl")

end # module ReactiveHMC
