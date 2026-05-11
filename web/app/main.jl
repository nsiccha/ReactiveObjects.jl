using Revise
using ReactiveObjectsWeb

begin
    ReactiveObjectsWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8103
    ReactiveObjectsWeb.serve(; host="0.0.0.0", revise=:lazy, port, async=true)
end
