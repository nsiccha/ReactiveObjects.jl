module ReactiveObjects

export @reactive, restore!, fcopy!, rcopy!

using OrderedCollections


isemutable(x) = isemutabletype(typeof(x))
isemutabletype(T) = ismutabletype(T) || (
    fieldcount(T) == 0 ? !isbitstype(T) : all(isemutabletype, fieldtypes(T))
)

@inline unval(x::Symbol) = x
@inline unval(::Val{T}) where {T} = T
struct ReactiveObject{S, D}
    valid::Vector{Bool}
    data::D
    @inline ReactiveObject(S, valid, data) = new{S, typeof(data)}(valid, data)
end
@inline sig(::ReactiveObject{S}) where {S} = S
@inline valid(obj::ReactiveObject) = getfield(obj, :valid)
@inline data(obj::ReactiveObject) = getfield(obj, :data)
Base.show(io::IO, obj::ReactiveObject) = begin 
    print(io, "ReactiveObject{", sig(obj), "}(\n")
    for (valid, (key, value)) in zip(valid(obj), pairs(data(obj)))
        # The printing of anonymous nodes, especially multiline ones, could be improved.
        printstyled(io, "    ", key, " (", valid ? "valid" : "invalid", ") = ", maybeunwrap(value), "\n"; color=valid ? :blue : :red)
    end
    print(io, ")")
end
@inline propertyidx(obj::ReactiveObject, x::Symbol) = propertyidx(obj, Val(x))
@inline isvalid(obj::ReactiveObject, x::Symbol) = isvalid(obj, Val(x))
@inline isvalid(obj::ReactiveObject, x::Val) = @inbounds valid(obj)[propertyidx(obj, x)]
@inline invalidatedependants!(obj::ReactiveObject, x::Symbol) = invalidatedependants!(obj, Val(x))
@inline compute!(obj::ReactiveObject, x::Symbol) = compute!(obj, Val(x))
"""
Restores a `ReactiveObject`'s valid state.
"""
function restore! end

@inline Base.getproperty(obj::ReactiveObject, x::Symbol) = Base.getproperty(obj, Val(x))
@inline Base.getproperty(obj::ReactiveObject, x::Val) = begin
    isvalid(obj, x) || begin
        # @debug "Recomputing $(unval(x))!"
        compute!(obj, x)
        # @debug "Recomputed $(unval(x))!"
    end
    maybeunwrap(getfield(data(obj), unval(x)))
end
@inline Base.setproperty!(obj::ReactiveObject, x::Symbol, val) = Base.setproperty!(obj, Val(x), val)
@inline Base.setproperty!(obj::ReactiveObject, x::Val, val) = begin
    rcopy!(getfield(data(obj), unval(x)), val)
    if isvalid(obj, x) 
        invalidatedependants!(obj, x)
    else
        @inbounds valid(obj)[propertyidx(obj, x)] = true
    end
    val
end
struct ReactiveProperty{S, O <: ReactiveObject}
    obj::O
    @inline ReactiveProperty(S, obj) = new{S, typeof(obj)}(obj)
end
@inline Base.parent(rp::ReactiveProperty) = rp.obj
@inline name(::ReactiveProperty{S}) where {S} = S
@inline Base.dotgetproperty(obj::ReactiveObject, x::Symbol) = ReactiveProperty(x, obj)
@inline Base.materialize!(dest::ReactiveProperty, bc::Base.Broadcast.Broadcasted{<:Any}) = begin 
    x, obj = name(dest), parent(dest)
    Base.materialize!(maybeunwrap(getfield(data(obj), unval(x))), bc)
    if isvalid(obj, x) 
        invalidatedependants!(obj, x)
    else
        @inbounds valid(obj)[propertyidx(obj, x)] = true
    end
end

@inline maybewrap(x) = isemutable(x) ? x : Ref(x)
@inline maybewrap(x::Union{NamedTuple,Tuple}) = map(maybewrap, x)
@inline maybeunwrap(x) = x
@inline maybeunwrap(x::Ref) = x[]
@inline maybeunwrap(x::Union{NamedTuple,Tuple}) = map(maybeunwrap, x)
@inline rcopy!(x::Union{NamedTuple,Tuple}, val) = map(rcopy!, x, val)
@inline rcopy!(x, val) = begin
    # @debug "WARNING: Copying $(typeof(x)) <= $(typeof(val))"
    copy!(x, val)
end
@inline rcopy!(::Function, ::Function) = nothing
@inline rcopy!(x::ReactiveObject, val::ReactiveObject) = begin
    rcopy!(valid(x), valid(val))
    rcopy!(data(x), data(val))
end
@inline rcopy!(x::Ref{T}, val::T) where {T} = (x[] = val)
@inline rcopy!(x::Ref{T}, val::Ref{T}) where {T} = (x[] = val[])
@inline fcopy!(dest, f, args...; kwargs...) = rcopy!(dest, f(args...; kwargs...))
@inline mymaterialize(x) = Base.materialize(x)

xeq(lhs, rhs) = Expr(:(=), lhs, rhs)
xeqinline(lhs, rhs) = :(@inline $lhs = $(xblock(rhs)))
xcall(args...; kwargs...) = Expr(
    :call, args[1], Expr(:parameters, [
        Expr(:kw, key, value) for (key, value) in pairs(kwargs)
    ]...), args[2:end]...
)#Expr(:call, args...)
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
dependency!(lhs::Symbol, rhs::Expr; info) = dependency!.(Ref(lhs), rhs.args; info)
dependency!(lhs::Expr, rhs::Expr; info) = dependency!.(Ref(lhs), rhs.args; info)
dependency!(lhs::Symbol, rhs; info) = begin
    lidx = get!(info.idxs, lhs) do
        push!(info.names, lhs)
        push!(info.dependants, Set{Int}())
        push!(info.dependencies, Set{Int}())
        push!(info.alldependants, Set{Int}())
        push!(info.alldependencies, Set{Int}())
        length(info.names)
    end
    if rhs in keys(info.idxs) 
        ridx = info.idxs[rhs]
        push!(info.dependants[ridx], lidx)
        push!(info.dependencies[lidx], ridx)
        push!(info.alldependencies[lidx], ridx)
        union!(info.alldependencies[lidx], info.alldependencies[ridx])
    end
end
dependency!(lhs::Expr, rhs; info) = begin 
    @assert Meta.isexpr(lhs, (:tuple, :parameters))
    dependency!.(lhs.args, Ref(rhs); info)
end
localize(x::Expr; info) = Expr(x.head, localize.(x.args; info)...)
localize(x::Symbol; info) = x in keys(info.idxs) ? :(DATA.$x) : (@assert x != :obj "What kind of madness makes you close over something named `obj`?"; x)
localize(x; info) = x
lhs_symbols(x::Vector) = mapreduce(lhs_symbols, vcat, x; init=Symbol[])
lhs_symbols(x::Symbol) = [x] 
lhs_symbols(x::Expr) = if x.head == :tuple
    lhs_symbols(x.args)
else
    error("Don't know how to handle $(x.head)")
end
coidxs!(i, lhs::Expr; info) = begin
    @assert Meta.isexpr(lhs, (:tuple, :parameters))
    for lhsi in lhs.args
        coidxs!(i, lhsi; info)
    end 
end
coidxs!(i, lhs::Symbol; info) = push!(info.coidxs[i], info.idxs[lhs])
xvalid(args...) = Symbol(:VALID, args...)
xdata(args...) = Symbol(:DATA, args...)
def!(target::Symbol, stmt; info) = if info.idxs[target] > length(info.restmts)
    i = info.idxs[target]
    lhs, rhs = stmt.args
    push!(info.coidxs, Set{Int}())
    coidxs!(i, lhs; info)
    coidxs = sort(collect(info.coidxs[i]))
    # @info target => lhs => coidxs => view(info.names, coidxs)
    V = xvalid()
    D = xdata()
    llhs = localize(lhs; info)
    restmt = if stmt.head == :(=) 
        if Meta.isexpr(rhs, :call)
            xcall(fcopy!, llhs, rhs.args...)
        elseif stmt.head == :(=) && Meta.isexpr(rhs, :do)
            pcall, c = rhs.args
            f, args... = pcall.args
            xcall(fcopy!, llhs, f, c, args...)
        else
            xcall(rcopy!, llhs, rhs)
        end
    else
        Expr(stmt.head, llhs, rhs)
    end
    push!(info.restmts, xblock(restmt, [:($V[$idx] = true) for idx in coidxs]...))
    @assert length(info.coidxs) == length(info.restmts) == i (length(info.restmts), i)
    alldependencies = sort(
        collect(info.alldependencies[i]); 
        lt=(j, k)->if k in info.coidxs[j] && j ∉ info.coidxs[k]
            true
        elseif j in info.coidxs[k] && k ∉ info.coidxs[j]
            false
        else
            j < k
        end
    )
    push!(info.defs, xeqinline(
        xcall(compute!, info.xobj, :(::Val{$(Meta.quot(target))})), 
        xblock(
            :($V = $valid(obj)),
            :($D = $data(obj)),
            [
                xblock(
                    :($V[$idx] || $restmt),
                    :($name = $maybeunwrap(getfield($D, $idx)))
                )
                for (idx, restmt, name) in zip(alldependencies, view(info.restmts, alldependencies), view(info.names, alldependencies))
            ]...,
            info.restmts[i],
        )
    ))
end
def!(lhs::Expr, stmt; info) = begin
    @assert Meta.isexpr(lhs, (:tuple, :parameters))
    for lhsi in lhs.args
        def!(lhsi, stmt; info)
    end 
end
method!(x; info) = x
method!(x::Symbol; info) = x in keys(info.idxs) ? :(obj.$x) : x
method!(x::Expr; info) = Expr(x.head, method!.(x.args; info)...)

# trueidxs(itr) = filter(Base.Fix1(getindex, itr), eachindex(itr))

reactive_expr(x::Expr; __module__) = begin
    V = xvalid()
    D = xdata()
    # We should allow "the other" way of defining functions - I prefer this way though
    @assert Meta.isexpr(x, :(=))
    lhs, rhs = x.args
    @assert Meta.isexpr(lhs, :call)
    f, args... = lhs.args
    # We should actually allow/encourage type annotations
    # The signature should/must actually include the types of the arguments
    sig = f
    xobj = :(obj::$ReactiveObject{$sig})
    names = arg_symbols(args)
    idxs = Dict([arg=>i for (i, arg) in enumerate(names)])
    dependants = [Set{Int}() for _ in names]
    dependencies = [Set{Int}() for _ in names]
    coidxs =  [Set{Int}() for _ in names]
    alldependants = [Set{Int}() for _ in names]
    alldependencies = [Set{Int}() for _ in names]
    # withpath = Set{Int}()
    restmts = Any[:($V[$idx] = true) for idx in eachindex(names)]
    defs = []
    info = (;xobj, names, idxs, dependants, dependencies, coidxs, alldependants, alldependencies, restmts, defs)
    @assert Meta.isexpr(rhs, :block)
    stmts = []
    methods = []
    for stmt in rhs.args
        if Meta.isexpr(stmt, (:(=))) && Meta.isexpr(stmt.args[1], :call)
            push!(methods, stmt)
            continue
        end
        push!(stmts, denode!(stmt; stmts))
    end 
    copy!(rhs.args, stmts)
    for (i, stmt) in enumerate(rhs.args)
        # We should really be doing something with the line number nodes!
        isa(stmt, LineNumberNode) && continue
        Meta.isexpr(stmt, :macrocall) && (stmt = macroexpand(__module__, stmt; recursive=false))
        # We should at least also allow docstrings!
        @assert Meta.isexpr(stmt, (:(=), :(.=)))
        slhs, srhs = stmt.args
        Meta.isexpr(stmt, :(.=)) && (rhs.args[i] = xeq(slhs, xcall(mymaterialize, srhs)) )
        dependency!(slhs, srhs; info)
        def!(slhs, stmt; info)
    end
    allalldependencies = sort(
        1:length(names); 
        lt=(j, k)->if k in info.coidxs[j] && j ∉ info.coidxs[k]
            true
        elseif j in info.coidxs[k] && k ∉ info.coidxs[j]
            false
        else
            j < k
        end
    )
    push!(info.defs, xeqinline(
        xcall(restore!, xobj; force=true), 
        xblock(
            :($V = $valid(obj)),
            :(force && ($V .= false)),
            :($D = $data(obj)),
            [
                xblock(
                    :($V[$idx] || $restmt),
                    :($name = $maybeunwrap(getfield($D, $idx)))
                )
                for (idx, restmt, name) in zip(allalldependencies, view(restmts, allalldependencies), view(names, allalldependencies))
            ]...,
        )
    ))
    for (i, ad) in enumerate(alldependencies), j in ad
        push!(alldependants[j], i)
    end
    for (i, lhs) in enumerate(names)
        vlhs = :(::Val{$(Meta.quot(lhs))})
        push!(defs, xeqinline(xcall(propertyidx, xobj, vlhs), i))
        push!(defs, xeqinline(
            xcall(invalidatedependants!, xobj, vlhs), 
            if length(alldependants[i]) == 0
                nothing
            else
                xblock(:(v = $valid(obj)), [:(v[$idx] = false) for idx in sort(collect(alldependants[i]))]...)
            end
        ))
    end
    push!(rhs.args, quote 
        return $ReactiveObject($sig, fill(true, $(length(names))), $maybewrap((;$(names...))))
    end)
    for method in methods
        lhs, rhs = method.args
        insert!(lhs.args, 2, xobj)
        push!(defs, xeq(lhs, method!(rhs; info)))
    end
    Expr(:block, x, defs...)
end

end # module ReactiveObjects
