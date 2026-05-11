module ReactiveObjectsWeb

using HTMXObjects
using TestModules

include("test/runtests.jl")

@htmx struct AppContext
    
    @get index = htmx(h.main(class="container")(
        h.h1("ReactiveObjectsWeb"),
        h.p("Edit src/ReactiveObjectsWeb.jl and Revise will reload automatically."),
        h.p(h.a(href="/tests")("Tests")),
    ); pico_version="2")

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
end

function __init__()
    route!(AppContext())
end

end # module ReactiveObjectsWeb
