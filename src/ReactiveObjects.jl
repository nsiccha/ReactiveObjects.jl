module ReactiveObjects

export @reactive

using OrderedCollections

struct ReactiveObject{S, D}
    valid::Vector{Bool}
    data::D
    ReactiveObject(S, valid, data) = new{S, typeof(data)}(valid, data)
end
sig(::ReactiveObject{S}) where {S} = S
valid(obj::ReactiveObject) = getfield(obj, :valid)
data(obj::ReactiveObject) = getfield(obj, :data)
Base.show(io::IO, obj::ReactiveObject) = begin 
    print(io, "ReactiveObject{", sig(obj), "}(\n")
    for (valid, (key, value)) in zip(valid(obj), pairs(data(obj)))
        # The printing of anonymous nodes, especially multiline ones, could be improved.
        printstyled(io, "    ", key, " (", valid ? "valid" : "invalid", ") = ", maybeunwrap(value), "\n"; color=valid ? :blue : :red)
    end
    print(io, ")")
end

Base.getproperty(obj::ReactiveObject, x::Symbol) = begin
    isvalid(obj, x) || compute!(obj, x)
    maybeunwrap(getfield(data(obj), x))
end
Base.setproperty!(obj::ReactiveObject, x::Symbol, val) = begin
    # @debug "Setting $x property."
    mycopy!(getfield(data(obj), x), val)
    if isvalid(obj, x) 
        invalidatedependants!(obj, x)
    else
        valid(obj)[propertyidx(obj, x)] = true
    end
    val
end
propertyidx(obj::ReactiveObject, x::Symbol) = propertyidx(obj, Val(x))
isvalid(obj::ReactiveObject, x::Symbol) = valid(obj)[propertyidx(obj, x)]
invalidate!(obj::ReactiveObject, x::Symbol) = if isvalid(obj, x) 
    valid(obj)[propertyidx(obj, x)] = false
    invalidatedependants!(obj, x)
end
invalidatedependants!(obj::ReactiveObject, x::Symbol) = invalidatedependants!(obj, Val(x))
compute!(obj::ReactiveObject, x::Symbol) = compute!(obj, Val(x))

maybewrap(x::Union{NamedTuple,Tuple}) = map(maybewrap, x)
# This is probably often fine, but not always
maybewrap(x::Function) = x
maybewrap(x::AbstractArray) = x
maybewrap(x) = ismutable(x) ? x : Ref(x)
maybeunwrap(x::Union{NamedTuple,Tuple}) = map(maybeunwrap, x)
maybeunwrap(x) = x
maybeunwrap(x::Ref) = x[]
mycopy!(x::Union{NamedTuple,Tuple}, val) = map(mycopy!, x, val)
mycopy!(x, val) = copy!(x, val)
mycopy!(x::Ref, val) = (x[] = val)

xeq(lhs, rhs) = Expr(:(=), lhs, rhs)
xcall(args...; kwargs...) = Expr(:call, args...)
xblock(args...) = Expr(:block, args...)
# I think there should be a better way to do this
xcall(f::Function, args...; kwargs...) = xcall(Expr(:., @__MODULE__, Meta.quot(Symbol(f))), args...; kwargs...)

macro reactive(x)
    esc(top_reactive_expr(x))
end
macro node(x)
    error("Macro @node doesn't work in a standalone context.")
end
arg_symbols(x::Vector) = mapreduce(arg_symbols, vcat, x; init=Symbol[])
arg_symbols(x::Symbol) = x
arg_symbols(x::Expr) = if Meta.isexpr(x,  (:(::), :(=), :kw))
    @assert length(x.args) == 2
    arg_symbols(x.args[1])
elseif x.head == :parameters
    arg_symbols(x.args)
else
    error("Don't know how to handle $x")
end
denode!(x::Expr; stmts) = if x.head == :macrocall && x.args[1] == Symbol("@node")
    @assert length(x.args) == 3
    _, lnn, node = x.args
    @assert lnn isa LineNumberNode
    snode = Symbol(node)
    push!(stmts, denode!(xeq(snode, node); stmts))
    snode
else
    Expr(x.head, denode!.(x.args; stmts)...)
end
denode!(x; stmts) = x
dependency!(lhs, rhs; dag) = nothing
dependency!(lhs, rhs::Expr; dag) = for rhsi in rhs.args
    dependency!(lhs, rhsi; dag)
end 
dependency!(lhs::Symbol, rhs::Symbol; dag) = begin
    get!(OrderedSet{Symbol}, dag, lhs)
    rhs in keys(dag) && push!(dag[rhs], lhs)
end
dependency!(lhs::Expr, rhs::Symbol; dag) = begin 
    @assert Meta.isexpr(lhs, :tuple)
    for lhsi in lhs.args
        dependency!(lhsi, rhs; dag)
    end 
end
localize(x::Expr; dag) = Expr(x.head, localize.(x.args; dag)...)
localize(x::Symbol; dag) = x in keys(dag) ? :(obj.$x) : (@assert x != :obj "What kind of madness makes you close over something named `obj`?"; x)
localize(x; dag) = x
def!(lhs::Symbol, stmt; info) = push!(info.defs, xeq(
    xcall(compute!, info.xobj, :(::Val{$(Meta.quot(lhs))})), 
    localize(stmt; info.dag)
))
def!(lhs::Expr, rhs; info) = begin 
    @assert Meta.isexpr(lhs, :tuple)
    for lhsi in lhs.args
        def!(lhsi, rhs; info)
    end 
end

top_reactive_expr(x::Expr) = begin
    # We should allow "the other" way of defining functions - I prefer this way though
    @assert Meta.isexpr(x, :(=))
    lhs, rhs = x.args
    @assert Meta.isexpr(lhs, :call)
    f, args... = lhs.args
    # We should actually allow/encourage type annotations
    dag = OrderedDict([arg=>OrderedSet{Symbol}() for arg in arg_symbols(args)])
    defs = []
    # The signature should/must actually include the types of the arguments
    sig = f
    info = (;xobj=:(obj::$ReactiveObject{$sig}), dag, defs)
    @assert Meta.isexpr(rhs, :block)
    # We should really be doing something with the line number nodes!
    # lnn = nothing
    stmts = []
    for stmt in rhs.args
        push!(stmts, denode!(stmt; stmts))
    end 
    # Is it actually fine to do this?
    empty!(rhs.args)
    append!(rhs.args, stmts)
    # rhs = xblock(stmts...)
    for stmt in rhs.args
        isa(stmt, LineNumberNode) && continue
        # We should at least also allow docstrings!
        @assert Meta.isexpr(stmt, :(=))
        slhs, srhs = stmt.args
        dependency!(slhs, srhs; dag)
        def!(slhs, stmt; info)
    end
    for (i, (lhs, deps)) in enumerate(pairs(dag))
        vlhs = :(::Val{$(Meta.quot(lhs))})
        push!(defs, xeq(
            xcall(propertyidx, info.xobj, vlhs), i
        ))
        push!(defs, xeq(
            xcall(invalidatedependants!, info.xobj, vlhs), 
            xblock([xcall(invalidate!, :obj, Meta.quot(dep)) for dep in deps]...)
        ))
    end
    n_locals = length(dag)
    push!(rhs.args, quote 
        return $ReactiveObject($sig, fill(true, $n_locals), $maybewrap((;$(keys(dag)...))))
    end)
    Expr(:block, x, defs...)
end

end # module ReactiveObjects
