module ReactiveObjectsWeb

using HTMXObjects
using TestModules

include("test/runtests.jl")

@htmx struct AppContext
    @get index() = HTMXObjects.pico_page(h.div(
        h.h1("ReactiveObjectsWeb"),
        h.p("Edit src/ReactiveObjectsWeb.jl and Revise will reload automatically."),
        h.p(
            h.a(href="/tests")("Tests"),
            " · ",
            h.a(href="/structure")("Structure"),
        ),
    ))

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
    @include structure = HTMXObjects.StructureRoutes(; root=AppContext)
end

function __init__()
    route!(AppContext())
end

end # module ReactiveObjectsWeb
