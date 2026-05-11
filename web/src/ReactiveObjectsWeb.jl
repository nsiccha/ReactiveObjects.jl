module ReactiveObjectsWeb

using HTMXObjects
using TestModules
using ReactiveObjects
# Treebars must be loaded before `using HTMXObjects: RecordingRoutes` so
# the HTMXObjectsTreebarsExt extension activates and gives the recording
# mount its polling progress UI. See htmxo-gallery §2c and §6.
using Treebars
using HTMXObjects: RecordingRoutes

include("test/runtests.jl")

# --- AppData ---
#
# Holds the gallery itself plus the recording paths. The `entry(id)`
# indexed inner struct caches per-item evaluation so each `.jl` body is
# `Base.include`d exactly once per AppData instance.

@dynamicstruct struct ReactiveObjectsAppData
    gallery_dir = joinpath(dirname(dirname(@__DIR__)), "web", "gallery")
    gallery     = Gallery(gallery_dir)

    recording_dir  = joinpath(dirname(dirname(@__DIR__)), "docs", "src", "public", "live-reactiveobjects")
    recording_base = get(ENV, "RECORD_BASE_PREFIX", "/ReactiveObjects.jl/dev/live-reactiveobjects")
    recording_paths = let ids = [it.id for it in gallery.items]
        ["/", "/gallery", ["/entries/$id" for id in ids]...]
    end

    # Per-id loaded demo + derived display values. The body of each
    # gallery file returns a NamedTuple `(; kernel, obj, fields, steps)`.
    # The `@reactive` macro defines `_demo_*` at module scope here when
    # the file is `Base.include`d into `@__MODULE__`.
    @struct entry(id) = begin
        item        = find_item(gallery, id)
        title       = item.title
        description = item.description
        code_string = item.code_string
        demo        = Base.include(@__MODULE__, item.path)
    end
end

const APPDATA = ReactiveObjectsAppData()

# --- Rendering helpers ---

# Render the valid/invalid state of every tracked field of a
# ReactiveObject. We don't read the fields here — that would trigger
# recomputation and defeat the demo. Instead we inspect `valid(obj)`
# directly.
function _state_row(obj, fields)
    v = ReactiveObjects.valid(obj)
    d = ReactiveObjects.data(obj)
    h.tr(
        [let
            idx = ReactiveObjects.propertyidx(obj, Val(f))
            is_valid = v[idx]
            value = is_valid ? repr(ReactiveObjects.maybeunwrap(getfield(d, idx))) : "—"
            h.td(; class=(is_valid ? "ro-valid" : "ro-invalid"))(
                h.code(string(f)), " = ", value,
            )
        end for f in fields]...,
    )
end

function render_demo(demo)
    fields = demo.fields
    obj    = demo.obj
    rows = []
    for step in demo.steps
        obj = step.op(obj)
        push!(rows, h.tr(h.th(step.label), [h.td() for _ in fields]...))
        push!(rows, _state_row(obj, fields))
    end
    h.div(
        h.table(; class="ro-trace")(
            h.thead(h.tr(h.th("step"), [h.th(h.code(string(f))) for f in fields]...)),
            h.tbody(rows...),
        ),
        h.p(h.small(
            "Each row shows the field values right after the labelled operation. ",
            "Em-dashes mark fields currently invalid (not yet recomputed since the last upstream write).",
        )),
    )
end

function reactiveobjects_card(item::GalleryItem, demo)
    h.article(
        h.header(h.h3(h.a(item.title; href="entries/$(item.id)"))),
        isempty(item.description) ? h.span() : h.p(item.description),
        render_demo(demo),
    )
end

# --- App ---

@htmx struct AppContext
    __appdata__ = APPDATA

    @get index() = HTMXObjects.pico_page(h.div(
        h.h1("ReactiveObjectsWeb"),
        h.p("A live demo of ", h.code("@reactive"), " kernels — dependency-driven lazy recomputation."),
        h.p(
            h.a("Gallery";        href=__self__/"gallery"),       " · ",
            h.a("Record gallery"; href=__self__/"record_gallery"),  " · ",
            h.a("Tests";          href=__self__/"tests"),         " · ",
            h.a("Structure";      href=__self__/"structure"),
        ),
    ))

    @get gallery() = let gallery = __appdata__.gallery
        h.div(
            [reactiveobjects_card(it, __appdata__.entry(it.id).demo) for it in gallery.items]...,
        )
    end

    @include entries(id::String) = begin
        e = __appdata__.entry(id)
        @get index() = h.article(
            h.header(h.h2(e.title)),
            isempty(e.description) ? h.span() : h.p(e.description),
            h.h4("Kernel"),
            h.pre(h.code(e.code_string; class="language-julia")),
            h.h4("Reactive trace"),
            render_demo(e.demo),
            h.p(h.a("← Back to gallery"; href=".."/"..")),
        )
    end

    @include record_gallery = RecordingRoutes(;
        app_type    = AppContext,
        paths       = __appdata__.recording_paths,
        record_dir  = __appdata__.recording_dir,
        record_base = __appdata__.recording_base,
        label       = "Recording ReactiveObjects gallery",
    )

    @include tests = TestRoutes(; __req__, test_module=@__MODULE__)
    @include structure = HTMXObjects.StructureRoutes(; root=AppContext)
end

function __init__()
    route!(AppContext())
end

end # module ReactiveObjectsWeb
