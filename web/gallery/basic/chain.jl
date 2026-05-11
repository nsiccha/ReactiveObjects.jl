# title: Chained dependencies
# description: A linear dependency chain x -> z -> w. Updating x invalidates both transitively; reading w recomputes z then w.
# section: basic
# tags: chain, invalidation, basic
# id: chain

@reactive _demo_chain(x) = begin
    z = 2x
    w = z + 1
end

(;
    kernel = _demo_chain,
    obj    = _demo_chain(3.0),
    fields = [:x, :z, :w],
    steps  = [
        (label = "initial",       op = obj -> obj),
        (label = "read .w",       op = obj -> (obj.w; obj)),
        (label = "set .x = 10",   op = obj -> (obj.x = 10.0; obj)),
        (label = "read .w again", op = obj -> (obj.w; obj)),
    ],
)
