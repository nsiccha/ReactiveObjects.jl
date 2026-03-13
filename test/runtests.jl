using Test, ReactiveObjects

# @reactive definitions must be at module scope — they use the function
# as a type parameter in ReactiveObject{S}, which requires a global binding.

@reactive test_basic(x, y) = begin
    z = x + y
    w = 2z
end

@reactive test_dep(x, y) = begin
    z = x + y
    w = z * 2
end

@reactive test_destruct(x) = begin
    a, b = (x, 2x)
    c = a + b
end

@reactive test_node(x) = begin
    y = @node(x^2) + 1
end

@reactive test_method(x) = begin
    y = 2x
    scale(__self__, factor) = factor * y
end

@reactive test_restore(x, y) = begin
    z = x + y
end

@reactive test_bcast(n) = begin
    v = zeros(n)
end

@reactive test_show(x) = begin
    y = 2x
end

@reactive test_chain(a) = begin
    b = a + 1
    c = b + 1
    d = c + 1
end

"""
    documented_kernel(x)

A documented kernel.
"""
@reactive documented_kernel(x) = begin
    y = x^2
end

@reactive prop_doc_kernel(x) = begin
    "The square of x"
    y = x^2
end

@reactive method_doc_kernel(x) = begin
    y = x^2
    "Double the value of y"
    double(__self__) = 2y
end

@reactive test_shared(x) = begin
    a = x + 1
    b = a * 2
    c = a * 3
end

@testset "ReactiveObjects.jl" begin

    @testset "Basic @reactive" begin
        obj = test_basic(3.0, 4.0)
        @test obj.x == 3.0
        @test obj.y == 4.0
        @test obj.z == 7.0
        @test obj.w == 14.0
    end

    @testset "Dependency invalidation" begin
        obj = test_dep(1.0, 2.0)
        @test obj.z == 3.0
        @test obj.w == 6.0

        obj.x = 10.0
        @test obj.z == 12.0
        @test obj.w == 24.0
    end

    @testset "Tuple destructuring" begin
        obj = test_destruct(3.0)
        @test obj.a == 3.0
        @test obj.b == 6.0
        @test obj.c == 9.0
    end

    @testset "@node subexpression" begin
        obj = test_node(3.0)
        @test obj.y == 10.0  # 3^2 + 1

        obj.x = 5.0
        @test obj.y == 26.0  # 5^2 + 1
    end

    @testset "Inline methods" begin
        obj = test_method(3.0)
        @test obj.y == 6.0
        @test scale(obj, 10.0) == 60.0

        obj.x = 5.0
        @test scale(obj, 10.0) == 100.0
    end

    @testset "restore!" begin
        obj = test_restore(1.0, 2.0)
        @test obj.z == 3.0

        obj.z = 99.0
        @test obj.z == 99.0

        restore!(obj; force=true)
        @test obj.z == 3.0
    end

    @testset "Broadcast assignment" begin
        obj = test_bcast(3)
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
        obj = test_show(3.0)
        s = sprint(show, obj)
        @test contains(s, "ReactiveObject")
        @test contains(s, "test_show")
    end

    @testset "Chained invalidation" begin
        obj = test_chain(0.0)
        @test obj.d == 3.0

        obj.a = 10.0
        @test obj.b == 11.0
        @test obj.c == 12.0
        @test obj.d == 13.0
    end

    @testset "Top-level docstring" begin
        doc = string(@doc documented_kernel)
        @test contains(doc, "documented kernel")
    end

    @testset "Inline property docstring" begin
        doc = string(@doc prop_doc_kernel)
        @test contains(doc, "square of x")
    end

    @testset "Inline method docstring" begin
        doc = string(@doc double)
        @test contains(doc, "Double the value")
    end

    @testset "Multiple fields with shared dependency" begin
        obj = test_shared(1.0)
        @test obj.b == 4.0
        @test obj.c == 6.0

        obj.x = 5.0
        @test obj.b == 12.0
        @test obj.c == 18.0
    end

end
