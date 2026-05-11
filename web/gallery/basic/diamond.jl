# title: Diamond dependency graph
# description: a -> b, a -> c, (b, c) -> d. Setting `a` invalidates all three downstream fields; reading `d` recomputes them in topo order with no redundant work.
# section: basic
# tags: diamond, dependency-graph
# id: diamond

@reactive _demo_diamond(a) = begin
    b = a + 1
    c = a + 2
    d = b + c
end

(;
    kernel = _demo_diamond,
    obj    = _demo_diamond(1.0),
    fields = [:a, :b, :c, :d],
    steps  = [
        (label = "initial",      op = obj -> obj),
        (label = "read .d",      op = obj -> (obj.d; obj)),
        (label = "set .a = 10",  op = obj -> (obj.a = 10.0; obj)),
        (label = "read .b only", op = obj -> (obj.b; obj)),
        (label = "read .d",      op = obj -> (obj.d; obj)),
    ],
)
