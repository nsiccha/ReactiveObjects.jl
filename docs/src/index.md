# ReactiveObjects.jl

`ReactiveObjects.jl` provides the `@reactive` macro for defining algorithmic kernels
that are clever about which parts get recomputed and when.

## Quick start

```julia
using ReactiveObjects

@reactive my_kernel(x, y) = begin
    z = f(x, y)       # z depends on x, y
    w = g(z)           # w depends on z (and transitively x, y)
end

obj = my_kernel(x_val, y_val)
obj.w          # returns precomputed w
obj.x = new_x  # invalidates z and w
obj.w          # recomputes z, then w
```

## Key concepts

### Reactive dependency tracking

The `@reactive` macro analyses the statements in the `begin` block at macro-expansion time
to build a dependency graph. When a field is set, all its dependants are invalidated.
When an invalidated field is read, it and its transitive dependencies are recomputed
in topological order.

### Tuple destructuring

A single expensive call can populate multiple fields:

```julia
@reactive phasepoint(grad_f, pos, mom) = begin
    pot, dpot_dpos = grad_f(pos)
    # pot and dpot_dpos are both set by one call to grad_f
end
```

When `compute!` is called for either `pot` or `dpot_dpos`, both are computed together.

### The `@node` macro

`@node` extracts a subexpression into its own named intermediate field for
finer-grained caching:

```julia
kin = .5 * (@node(logdet(chol_metric)) + dot(mom, dkin_dmom))
```

Here `logdet(chol_metric)` becomes a separate cached field that is only recomputed
when `chol_metric` changes.

### Broadcast assignment

Use `@.` to update array-valued fields in-place while triggering invalidation:

```julia
@. phasepoint.mom = mom0 - .5 * stepsize * phasepoint.dham_dpos
```

### Inline methods

Functions can be defined inside a `@reactive` block:

```julia
@reactive my_kernel(x, y) = begin
    z = f(x, y)
    my_method(__self__, extra_arg) = do_something(z, extra_arg)
end
```

`__self__` refers to the reactive object. Bare field names in the method body are
rewritten to `__self__.field`.

## Example: Riemannian HMC phase point

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

When used with a generalized leapfrog integrator, the reactive system avoids
redundant gradient and metric evaluations — only recomputing what actually changed.
