# ReactiveObjects.jl (soon to be) ReactiveKernels.jl

Provides reactive algorithmic kernels, i.e. ones that are clever about which parts get recomputed how and when.

For example, a kernel representing the phase point of the Riemannian HMC method would look like this:

```julia
@reactive riemannian_phasepoint(pot_f, grad_f, metric_f, metric_grad_f, pos, mom) = begin
    pot = pot_f(pos)
    pot, dpot_dpos = grad_f(pos)
    pot, dpot_dpos, metric = metric_f(pos)
    pot, dpot_dpos, metric, metric_grad = metric_grad_f(pos)

    chol_metric = cholesky(metric)
    inv_metric = Symmetric(inv(chol_metric))
    dkin_dmom = chol_metric \ mom

    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))

    dkin_dpos .= @node(map(eachslice(metric_grad; dims=3)) do pgi
        .5 * tr_prod(inv_metric, pgi)
    end) .- Base.broadcasted(eachslice(metric_grad; dims=3)) do pgi
        .5 * dot(dkin_dmom, pgi, dkin_dmom)
    end

    ham = pot + kin
    @. dham_dpos = dkin_dpos + dpot_dpos
    dham_dmom = dkin_dmom
end
```

Depending on which quantities are required when, the kernel will recompute e.g. the gradients in a clever (enough) way. 
Consider e.g. the interaction with the generalized leapfrog integrator below:

```julia
generalized_leapfrog!(phasepoint; stepsize, n_fi_steps) = begin 
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps 
        @. phasepoint.mom = mom0 - .5 * stepsize * phasepoint.dham_dpos
    end
    dham_dmom0 = copy(phasepoint.dham_dmom)
    for _ in 1:n_fi_steps 
        @. phasepoint.pos = pos0 + .5 * stepsize * (dham_dmom0 + phasepoint.dham_dmom)
    end
    @. phasepoint.mom -= .5 * stepsize * phasepoint.dham_dpos
end
```