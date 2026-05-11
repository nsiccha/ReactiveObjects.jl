using TestModules, ReactiveObjects

# --- @reactive definitions at module scope ---

@reactive _test_basic(x, y) = begin
    z = x + y
    w = 2z
end

@reactive _test_dep(x, y) = begin
    z = x + y
    w = z * 2
end

@reactive _test_destruct(x) = begin
    a, b = (x, 2x)
    c = a + b
end

@reactive _test_node(x) = begin
    y = @node(x^2) + 1
end

@reactive _test_method(x) = begin
    y = 2x
    _ro_scale(__self__, factor) = factor * y
end

@reactive _test_restore(x, y) = begin
    z = x + y
end

@reactive _test_bcast(n) = begin
    v = zeros(n)
end

@reactive _test_show(x) = begin
    y = 2x
end

@reactive _test_chain(a) = begin
    b = a + 1
    c = b + 1
    d = c + 1
end

"""
    _documented_kernel(x)

A documented kernel.
"""
@reactive _documented_kernel(x) = begin
    y = x^2
end

@reactive _prop_doc_kernel(x) = begin
    "The square of x"
    y = x^2
end

@reactive _method_doc_kernel(x) = begin
    y = x^2
    "Double the value of y"
    _ro_double(__self__) = 2y
end

@reactive _test_shared(x) = begin
    a = x + 1
    b = a * 2
    c = a * 3
end

@reactive _test_rcopy_nt(x) = begin
    nt = (a=x, b=2x)
end

@reactive _test_isemutable(x) = begin
    y = [x, 2x]
end

@reactive _test_diamond(a) = begin
    b = a + 1
    c = a + 2
    d = b + c
end

@reactive _test_setvalid(x) = begin
    y = 2x
end

# --- Tests ---

@testset "Basic @reactive" begin
    obj = _test_basic(3.0, 4.0)
    @test obj.x == 3.0
    @test obj.y == 4.0
    @test obj.z == 7.0
    @test obj.w == 14.0
end

@testset "Dependency invalidation" begin
    obj = _test_dep(1.0, 2.0)
    @test obj.z == 3.0
    @test obj.w == 6.0

    obj.x = 10.0
    @test obj.z == 12.0
    @test obj.w == 24.0
end

@testset "Tuple destructuring" begin
    obj = _test_destruct(3.0)
    @test obj.a == 3.0
    @test obj.b == 6.0
    @test obj.c == 9.0
end

@testset "@node subexpression" begin
    obj = _test_node(3.0)
    @test obj.y == 10.0

    obj.x = 5.0
    @test obj.y == 26.0
end

@testset "Inline methods" begin
    obj = _test_method(3.0)
    @test obj.y == 6.0
    @test _ro_scale(obj, 10.0) == 60.0

    obj.x = 5.0
    @test _ro_scale(obj, 10.0) == 100.0
end

@testset "restore!" begin
    obj = _test_restore(1.0, 2.0)
    @test obj.z == 3.0

    obj.z = 99.0
    @test obj.z == 99.0

    restore!(obj; force=true)
    @test obj.z == 3.0
end

@testset "Broadcast assignment" begin
    obj = _test_bcast(3)
    @test obj.v == [0.0, 0.0, 0.0]

    @. obj.v = [1.0, 2.0, 3.0]
    @test obj.v == [1.0, 2.0, 3.0]
end

@testset "rcopy! basics" begin
    a = [1.0, 2.0]
    b = [0.0, 0.0]
    rcopy!(b, a)
    @test b == a

    r = Ref(0.0)
    rcopy!(r, 5.0)
    @test r[] == 5.0
end

@testset "fcopy!" begin
    dest = [0.0, 0.0]
    fcopy!(dest, x -> 2 .* x, [1.0, 2.0])
    @test dest == [2.0, 4.0]
end

@testset "ReactiveObject display" begin
    obj = _test_show(3.0)
    s = sprint(show, obj)
    @test contains(s, "ReactiveObject")
    @test contains(s, "_test_show")
end

@testset "Chained invalidation" begin
    obj = _test_chain(0.0)
    @test obj.d == 3.0

    obj.a = 10.0
    @test obj.b == 11.0
    @test obj.c == 12.0
    @test obj.d == 13.0
end

@testset "Top-level docstring" begin
    doc = string(@doc _documented_kernel)
    @test contains(doc, "documented kernel")
end

@testset "Inline property docstring" begin
    doc = string(@doc _prop_doc_kernel)
    @test contains(doc, "square of x")
end

@testset "Inline method docstring" begin
    doc = string(@doc _ro_double)
    @test contains(doc, "Double the value")
end

@testset "Multiple fields with shared dependency" begin
    obj = _test_shared(1.0)
    @test obj.b == 4.0
    @test obj.c == 6.0

    obj.x = 5.0
    @test obj.b == 12.0
    @test obj.c == 18.0
end

@testset "rcopy! NamedTuple" begin
    obj = _test_rcopy_nt(3.0)
    @test obj.nt.a == 3.0
    @test obj.nt.b == 6.0

    r = Ref(1.0)
    rcopy!(r, Ref(5.0))
    @test r[] == 5.0
end

@testset "rcopy! Function no-op" begin
    @test rcopy!(sin, cos) === nothing
end

@testset "isemutable" begin
    @test ReactiveObjects.isemutable([1, 2, 3]) == true
    @test ReactiveObjects.isemutable(1.0) == false
    @test ReactiveObjects.isemutable(Ref(1.0)) == true
end

@testset "Diamond dependency graph" begin
    obj = _test_diamond(1.0)
    @test obj.d == 5.0

    obj.a = 10.0
    @test obj.d == 23.0
end

@testset "setproperty! on invalid field marks valid" begin
    obj = _test_setvalid(3.0)
    @test obj.y == 6.0

    ReactiveObjects.valid(obj)[ReactiveObjects.propertyidx(obj, Val(:y))] = false
    obj.y = 99.0
    @test obj.y == 99.0
end
