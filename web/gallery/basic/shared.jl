# title: Shared dependency reused across fields
# description: Two outputs (`b` and `c`) both derive from a single shared intermediate `a`. After setting `x`, reading `b` warms `a`; reading `c` reuses the cached `a` without recomputing it.
# section: basic
# tags: shared, caching
# id: shared

@reactive _demo_shared(x) = begin
    a = x + 1
    b = a * 2
    c = a * 3
end

(;
    kernel = _demo_shared,
    obj    = _demo_shared(1.0),
    fields = [:x, :a, :b, :c],
    steps  = [
        (label = "initial",     op = obj -> obj),
        (label = "set .x = 5",  op = obj -> (obj.x = 5.0; obj)),
        (label = "read .b",     op = obj -> (obj.b; obj)),
        (label = "read .c",     op = obj -> (obj.c; obj)),
    ],
)
