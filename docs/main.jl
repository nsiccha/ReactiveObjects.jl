using ReactiveObjects, LinearAlgebra
begin

@reactive euclidean_phasepoint(pot_f, metric, pos, mom) = begin 
    pot, dpot_dpos = pot_f(pos)

    chol_metric = cholesky(metric)
    dkin_dmom = chol_metric \ mom
    # The logdet term could be left out as it doesn't change (unless the metric changes)
    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))

    ham = pot + kin
    dham_dpos = dpot_dpos
    dham_dmom = dkin_dmom
end

leapfrog!(phasepoint; stepsize) = begin 
    phasepoint.mom -= @. .5 * stepsize * phasepoint.dham_dpos
    phasepoint.pos += @. stepsize * phasepoint.dham_dmom
    phasepoint.mom -= @. .5 * stepsize * phasepoint.dham_dpos
end

pot_f(pos) = (.5sum(abs2, pos), pos)
dim = 10
metric = Diagonal(ones(dim))
pos = randn(dim)
mom = randn(dim)

@info "Fully initialized QOIs"
obj = euclidean_phasepoint(pot_f, metric, pos, mom)
display(obj)

@info "Automatically invalidated QOIs after modifying `pos`"
obj.pos = randn(dim)
display(obj)

@info "Automatically recomputed `ham` ($(obj.ham))"
display(obj)

@info "After one leapfrog step"
leapfrog!(obj; stepsize=.1)
display(obj)

end

begin 

tr_prod(A::AbstractMatrix, B::AbstractMatrix) = sum(Base.broadcasted(*, A', B))

@reactive riemannian_phasepoint(pot_f, metric_f, pos, mom) = begin
    pot, dpot_dpos = pot_f(pos)

    # It would be better to have the interface be to compute (pot, dpot_dpos, metric, metric_grad) in one swoop
    metric, metric_grad = metric_f(pos)
    chol_metric = cholesky(metric)
    inv_metric = Symmetric(inv(chol_metric))
    dkin_dmom = chol_metric \ mom
    kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
    dkin_dpos = @node(map(eachindex(pos)) do i
        .5 * tr_prod(inv_metric, metric_grad[:, :, i])
    end) .- Base.broadcasted(eachindex(pos)) do i
        .5 * dot(dkin_dmom, metric_grad[:, :, i], dkin_dmom)
    end

    ham = pot + kin
    dham_dpos = dpot_dpos + dkin_dpos
    dham_dmom = dkin_dmom
end
generalized_leapfrog!(phasepoint; stepsize, n_fi_steps) = begin 
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps 
        phasepoint.mom = @. mom0 - .5 * stepsize * phasepoint.dham_dpos
    end
    dham_dmom0 = copy(phasepoint.dham_dmom)
    for _ in 1:n_fi_steps 
        phasepoint.pos = @. pos0 + .5 * stepsize * (dham_dmom0 + phasepoint.dham_dmom)
    end
    phasepoint.mom -= @. .5 * stepsize * phasepoint.dham_dpos
end

implicit_midpoint!(phasepoint; stepsize, n_fi_steps) = begin 
    pos0, mom0 = map(copy, (phasepoint.pos, phasepoint.mom))
    for _ in 1:n_fi_steps 
        (;dham_dmom, dham_dpos) = phasepoint
        phasepoint.pos = @. pos0 + .5 * stepsize * dham_dmom
        phasepoint.mom = @. mom0 - .5 * stepsize * dham_dpos
    end
    phasepoint.pos = @. 2 * phasepoint.pos - pos0
    phasepoint.mom = @. 2 * phasepoint.mom - mom0
end

metric_f(pos) = begin
    dim = length(pos)
    Diagonal(ones(dim)), zeros((dim, dim, dim))
end 

@info "Fully initialized QOIs"
obj = riemannian_phasepoint(pot_f, metric_f, pos, mom)
display(obj)

@info "Automatically invalidated QOIs after modifying `pos`"
obj.pos = randn(dim)
display(obj)

@info "Automatically recomputed `ham` ($(obj.ham))"
display(obj)

@info "After one generalized leapfrog step"
generalized_leapfrog!(obj; stepsize=.1, n_fi_steps=1)
display(obj)

@info "After one implicit midpoint step"
implicit_midpoint!(obj; stepsize=.1, n_fi_steps=1)
display(obj)

end

begin 

softabs(premetric; alpha=20.) = begin 
    premetric_eigvals, eigvecs = eigen(premetric)
    metric_eigvals = premetric_eigvals .* coth.(alpha * premetric_eigvals)
    metric_eigvals, premetric_eigvals, eigvecs
end

@reactive riemannian_softabs_phasepoint(pot_f, premetric_f, pos, mom; alpha=20.) = begin 
    pot, dpot_dpos = pot_f(pos)

    # It would be better to have the interface be to compute (pot, dpot_dpos, premetric, premetric_grad) in one swoop, as 
    # the premetric will "usually" be the hessian computed via AD 
    premetric, premetric_grad = premetric_f(pos)
    metric_eigvals, premetric_eigvals, Q = softabs(premetric; alpha)
    Q_inv = Q / Diagonal(metric_eigvals)

    dkin_dmom = Q_inv * (Q' * mom)
    kin = .5 * (@node(sum(log, metric_eigvals)) + dot(mom, dkin_dmom))
    J = broadcast(premetric_eigvals, metric_eigvals, premetric_eigvals', metric_eigvals') do pei, ei, pej, ej
        if pei == pej
            coth(alpha * pei) + pei * alpha * -csch(pei * alpha)^2 
        else
            ((ei - ej) / (pei - pej)) 
        end
    end
    D = Diagonal(Q' * mom)
    dkin_dpos = @node(map(1:dim) do i
        .5 * tr_prod(@node(Q_inv * Diagonal(view(J, diagind(J))) * Q'), premetric_grad[:, :, i])
    end) - @node(map(1:dim) do i
        # I wonder how clever Julia is about computing the below matrix product
        .5 * tr_prod(@node(Q_inv * (D * J * D) * Q_inv'), premetric_grad[:, :, i])
    end)

    ham = pot + kin
    dham_dpos = dpot_dpos + dkin_dpos
    dham_dmom = dkin_dmom
end


@info "Fully initialized QOIs"
obj = riemannian_softabs_phasepoint(pot_f, metric_f, pos, mom)
display(obj)

@info "Automatically invalidated QOIs after modifying `pos`"
obj.pos = randn(dim)
display(obj)

@info "Automatically recomputed `ham` ($(obj.ham))"
display(obj)

@info "After one generalized leapfrog step"
generalized_leapfrog!(obj; stepsize=.1, n_fi_steps=1)
display(obj)

@info "After one implicit midpoint step"
implicit_midpoint!(obj; stepsize=.1, n_fi_steps=1)
display(obj)

end