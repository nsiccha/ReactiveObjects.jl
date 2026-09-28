using Documenter, DocumenterVitepress, ReactiveObjects

# Note: the theme files under `src/.vitepress/theme/` (`htmxo-embed.ts`,
# `htmxo-gallery.css`, `htmxo-syntax.css`) are a committed snapshot synced
# from HTMXObjects@devibe c9a1bdcd1c5933fa936c409646278ca0c3bf3f18 —
# `make.jl` used to call `HTMXObjects.vitepress_theme_install(...)` here to
# auto-sync them, but that pinned the whole docs env on HTMXObjects'
# unregistered closure (HTMX was never registered, so every Docs run since
# May failed at resolve). Re-sync the snapshot manually from
# `HTMXObjects/assets/vitepress/` when the upstream embed runtime changes.

makedocs(
    sitename = "ReactiveObjects.jl",
    modules  = [ReactiveObjects],
    format   = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/nsiccha/ReactiveObjects.jl",
        devurl = "dev",
        devbranch = "dev",
    ),
    pages = [
        "Home"      => "index.md",
        "Gallery"   => "gallery.md",
        "API"       => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

# Ensure a root index.html redirect exists for when no stable version is deployed
let redirect = joinpath(@__DIR__, "build", "index.html")
    isfile(redirect) || write(redirect, """
    <!DOCTYPE html>
    <html><head>
    <meta http-equiv="refresh" content="0; url=dev/">
    </head><body>Redirecting to <a href="dev/">dev</a>...</body></html>
    """)
end

DocumenterVitepress.deploydocs(
    repo = "github.com/nsiccha/ReactiveObjects.jl",
    devbranch = "dev",
    push_preview = true,
)
