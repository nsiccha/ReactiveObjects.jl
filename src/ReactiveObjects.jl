module ReactiveObjects

export @reactive

using OrderedCollections

struct ReactiveObject{S, D}
    valid::Vector{Bool}
    data::D
    @inline ReactiveObject(S, valid, data) = new{S, typeof(data)}(valid, data)
end
struct Unreactive{O}
    obj::O
end
@inline Base.parent(obj::Unreactive) = obj.obj
@inline sig(::ReactiveObject{S}) where {S} = S
@inline valid(obj::ReactiveObject) = getfield(obj, :valid)
@inline valid(obj::Unreactive) = valid(parent(obj))
@inline data(obj::ReactiveObject) = getfield(obj, :data)
@inline data(obj::Unreactive) = data(parent(obj))
Base.show(io::IO, obj::ReactiveObject) = begin 
    print(io, "ReactiveObject{", sig(obj), "}(\n")
    for (valid, (key, value)) in zip(valid(obj), pairs(data(obj)))
        # The printing of anonymous nodes, especially multiline ones, could be improved.
        printstyled(io, "    ", key, " (", valid ? "valid" : "invalid", ") = ", maybeunwrap(value), "\n"; color=valid ? :blue : :red)
    end
    print(io, ")")
end

@inline unval(x::Symbol) = x
@inline unval(::Val{T}) where {T} = T
@inline Base.getproperty(obj::ReactiveObject, x::Symbol) = Base.getproperty(obj, Val(x))
@inline Base.getproperty(obj::ReactiveObject, x::Val) = begin
    isvalid(obj, x) || begin
        @debug "Recomputing $(unval(x))!"
        compute!(obj, x)
        @debug "Recomputed $(unval(x))!"
    end
    maybeunwrap(getfield(data(obj), unval(x)))
end
@inline Base.setproperty!(obj::ReactiveObject, x::Symbol, val) = Base.setproperty!(obj, Val(x), val)
@inline Base.setproperty!(obj::ReactiveObject, x::Val, val) = begin
    # @debug "Setting $x property."
    mycopy!(getfield(data(obj), unval(x)), val)
    if isvalid(obj, x) 
        invalidatedependants!(obj, x)
    else
        @inbounds valid(obj)[propertyidx(obj, x)] = true
    end
    val
end
@inline Base.setproperty!(uobj::Unreactive, x::Symbol, val) = Base.setproperty!(uobj, Val(x), val)
@inline Base.setproperty!((;obj)::Unreactive, x::Val, val) = begin
    mycopy!(getfield(data(obj), unval(x)), val)
    @inbounds valid(obj)[propertyidx(obj, x)] = true
    val
end
struct ReactiveProperty{S, O <: ReactiveObject}
    obj::O
    @inline ReactiveProperty(S, obj) = new{S, typeof(obj)}(obj)
end
@inline name(::ReactiveProperty{S}) where {S} = S
@inline Base.dotgetproperty(obj::ReactiveObject, x::Symbol) = ReactiveProperty(x, obj)
@inline Base.dotgetproperty((;obj)::Unreactive, x::Symbol) = ReactiveProperty(x, obj)
@inline Base.materialize!(dest::ReactiveProperty, bc::Base.Broadcast.Broadcasted{<:Any}) = begin 
    obj, x = dest.obj, name(dest)
    Base.materialize!(maybeunwrap(getfield(data(obj), unval(x))), bc)
    # Base.materialize!(Base.getproperty(obj, unval(x)), bc)
    if isvalid(obj, x) 
        invalidatedependants!(obj, x)
    else
        @inbounds valid(obj)[propertyidx(obj, x)] = true
    end
end
@inline propertyidx(obj::ReactiveObject, x::Symbol) = propertyidx(obj, Val(x))
@inline isvalid(obj::ReactiveObject, x::Symbol) = isvalid(obj, Val(x))
@inline invalidate!(obj::ReactiveObject, x::Symbol) = invalidate!(obj, Val(x))
@inline isvalid(obj::ReactiveObject, x::Val) = @inbounds valid(obj)[propertyidx(obj, x)]
@inline invalidate!(obj::ReactiveObject, x::Val) = if isvalid(obj, x) 
    @inbounds valid(obj)[propertyidx(obj, x)] = false
    invalidatedependants!(obj, x)
end
@inline invalidatedependants!(obj::ReactiveObject, x::Symbol) = invalidatedependants!(obj, Val(x))
@inline compute!(obj::ReactiveObject, x::Symbol) = compute!(obj, Val(x))

@inline maybewrap(x::Union{NamedTuple,Tuple}) = map(maybewrap, x)
# This is probably often fine, but not always
@inline maybewrap(x::Function) = x
@inline maybewrap(x::AbstractArray) = x
@inline maybewrap(x) = ismutable(x) ? x : Ref(x)
@inline maybeunwrap(x::Union{NamedTuple,Tuple}) = map(maybeunwrap, x)
@inline maybeunwrap(x) = x
@inline maybeunwrap(x::Ref) = x[]
@inline mycopy!(x::Union{NamedTuple,Tuple}, val) = map(mycopy!, x, val)
@inline mycopy!(x, val) = copy!(x, val)
@inline mycopy!(x::Ref, val) = (x[] = val)
@inline mymaterialize(x) = Base.materialize(x)

xeq(lhs, rhs) = Expr(:(=), lhs, rhs)
xeqinline(lhs, rhs) = :(@inline $lhs = $(xblock(rhs)))
xcall(args...; kwargs...) = Expr(:call, args...)
xblock(args...) = Expr(:block, args...)
# I think there should be a better way to do this
xcall(f::Function, args...; kwargs...) = xcall(Expr(:., @__MODULE__, Meta.quot(Symbol(f))), args...; kwargs...)

macro reactive(x)
    esc(reactive_expr(x; __module__))
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
dependency!(lhs, rhs; info) = nothing
dependency!(lhs, rhs::Expr; info) = for rhsi in rhs.args
    dependency!(lhs, rhsi; info)
end 
dependency!(lhs::Symbol, rhs::Symbol; info) = begin
    get!(OrderedSet{Symbol}, info.dag, lhs)
    get!(OrderedSet{Symbol}, info.rdag, lhs)
    if rhs in keys(info.dag) 
        push!(info.dag[rhs], lhs)
        push!(info.rdag[lhs], rhs)
    end
end
dependency!(lhs::Expr, rhs::Symbol; info) = begin 
    @assert Meta.isexpr(lhs, (:tuple, :parameters))
    for lhsi in lhs.args
        dependency!(lhsi, rhs; info)
    end 
end
localize(x::Expr; dag) = Expr(x.head, localize.(x.args; dag)...)
localize(x::Symbol; dag) = x in keys(dag) ? :(uobj.$x) : (@assert x != :obj "What kind of madness makes you close over something named `obj`?"; x)
localize(x; dag) = x
def!(lhs::Symbol, stmt; info) = if lhs ∉ info.withpath
    push!(info.defs, xeqinline(
        xcall(compute!, info.xobj, :(::Val{$(Meta.quot(lhs))})), 
        xblock(
            :((;$(info.rdag[lhs]...)) = obj),
            :(uobj = $Unreactive(obj)), 
            Expr(stmt.head, localize(stmt.args[1]; info.dag), stmt.args[2])
        )
    ))
    push!(info.withpath, lhs)
end
def!(lhs::Expr, stmt; info) = begin 
    @assert Meta.isexpr(lhs, (:tuple, :parameters))
    for lhsi in lhs.args
        def!(lhsi, stmt; info)
    end 
end

reactive_expr(x::Expr; __module__) = begin
    # We should allow "the other" way of defining functions - I prefer this way though
    @assert Meta.isexpr(x, :(=))
    lhs, rhs = x.args
    @assert Meta.isexpr(lhs, :call)
    f, args... = lhs.args
    # We should actually allow/encourage type annotations
    dag = OrderedDict([arg=>OrderedSet{Symbol}() for arg in arg_symbols(args)])
    rdag = empty(dag)
    defs = []
    # The signature should/must actually include the types of the arguments
    sig = f
    withpath = Set{Symbol}()
    info = (;xobj=:(obj::$ReactiveObject{$sig}), dag, rdag, defs, withpath)
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
    for (i, stmt) in enumerate(rhs.args)
        isa(stmt, LineNumberNode) && continue
        Meta.isexpr(stmt, :macrocall) && (stmt = macroexpand(__module__, stmt; recursive=false))
        # We should at least also allow docstrings!
        @assert Meta.isexpr(stmt, (:(=), :(.=)))
        slhs, srhs = stmt.args
        Meta.isexpr(stmt, :(.=)) && (rhs.args[i] = xeq(slhs, xcall(mymaterialize, srhs)) )
        dependency!(slhs, srhs; info)
        def!(slhs, stmt; info)
    end
    for (i, (lhs, deps)) in enumerate(pairs(dag))
        vlhs = :(::Val{$(Meta.quot(lhs))})
        push!(defs, xeqinline(
            xcall(propertyidx, info.xobj, vlhs), i
        ))
        push!(defs, xeqinline(
            xcall(invalidatedependants!, info.xobj, vlhs), 
            xblock([xcall(invalidate!, :obj, Val(dep)) for dep in deps]...)
        ))
    end
    n_locals = length(dag)
    push!(rhs.args, quote 
        return $ReactiveObject($sig, fill(true, $n_locals), $maybewrap((;$(keys(dag)...))))
    end)
    Expr(:block, x, defs...)
end

end # module ReactiveObjects
