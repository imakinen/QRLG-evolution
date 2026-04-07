import LinearAlgebra as LA
import ExponentialUtilities: expv
import MatrixMarket: mmread
import SparseArrays: SparseMatrixCSC
import ArgParse
import Dates
import Logging
import LoggingExtras
import Printf: @sprintf
import Serialization: serialize, deserialize


struct Operator{T<:Number}
    name::String
    j_max::Integer
    matrix::SparseMatrixCSC{T, Int64}
end


function Operator(name, j_max, args...)
    msg = "loading operator $name"
    if name == "H_abc"
        kind, a, b, c = args
        msg = msg * " (kind = $(args[1])," * 
                    " a = $(args[2]), b = $(args[3]), c = $(args[4]))"
    end
    @info msg
    path = "data/julia/$(j_max)/operators/$name.mtx"
    if !isfile(path)
        @info "Operator not found. Running convert.py"
        cmd = `python convert.py $name $j_max $(args)`
        io = IOBuffer()
        process = run(pipeline(ignorestatus(cmd), stdout=io, stderr=io))
        wait(process)
        output = String(take!(io))
        if process.exitcode != 0
            msg = "convert.py terminated due to error"
            @error "$msg\n$output"
            exit(process.exitcode)
        end
    end
    matrix = mmread(path)
    T = eltype(matrix)
    return Operator{T}(name, j_max, matrix)
end


mutable struct CoherentState
    j_0::Integer
    c_0::Float64
    t::Float64
    j_max::Integer
    vector::Vector{ComplexF64}
end


function CoherentState(j_0, c_0, t, j_max)

    function amplitude(j, j_0, c_0, t)
        return sqrt(2*j + 1) * exp(-t * (j - j_0)^2 / 2) * exp(-im * c_0 * j)
    end

    v = [amplitude(j, j_0, c_0, t) for j in 1:j_max]
    v = v / LA.norm(v)
    psi = kron(v, kron(v, v))
    return CoherentState(j_0, c_0, t, j_max, psi)
end


function evolve!(state, H, dt, m)
    state.vector = expv(dt, -im * H, state.vector, m=m)
end


function write_ev(operator, state, t, filename;
                  T=nothing, steps=nothing, m=nothing)
    if t == 0.0
        open(filename, "w") do f
            write(f, "# Operator: $(operator.name)\n")
            write(f, "# j_0 = $(state.j_0), c_0 = $(state.c_0), " *
                     "t = $(state.t), j_max = $(state.j_max), " *
                     "T = $T, steps = $steps, m = $m\n")
        end
    end
    A = operator.matrix
    psi_t = state.vector
    Apsi_t = A * psi_t
    A_t = real(LA.dot(psi_t, Apsi_t))
    A2_t = LA.norm(Apsi_t)^2
    open(filename, "a") do f
        write(f, "$t $A_t $A2_t\n")
    end
end


function write_norm(state, t, filename)
    open(filename, "a") do f
        write(f, "$t $(LA.norm(state.vector))\n")
    end
end


function evolution(state, H, T, steps, m;
                   operators=nothing, model="", restore=false)

    @info "j_0 = $(state.j_0), c_0 = $(state.c_0), t = $(state.t); " *
          "j_max = $(state.j_max), T = $T, steps = $steps, m = $m. " *
          "Model: $(model)"

    parameters = Any[state.j_0, state.c_0, state.t, state.j_max, T, steps]
    parameters_str = join(string.(parameters) .|> x -> x[1:min(8, end)], "__")
    directory = "data/julia/$(state.j_max)/$(model)/$(parameters_str)/m=$m"
    mkpath(directory)
    mkpath("states")

    if operators === nothing
        operators = ["p_x", "p_z", "c_x", "c_z", "s_x", "s_z",
                     "V", "V_inv", "N_jmin", "N_jmax", "N_jmax-1"]
    end

    if restore
        open("states/$(parameters_str)__$m.bin", "r") do f
            t, state = deserialize(f)
        end
        @info "Restoring from last saved state. t = $t"
    else
        t = 0.0
    end
    @info "t = $t"

    operators = Operator.(operators, state.j_max)

    write_norm(state, t, "$directory/norm.txt")
    write_ev(H, state, t, "$directory/H.txt"; T, steps, m)

    for operator in operators
        filename = "$directory/$(operator.name).txt"
        write_ev(operator, state, t, filename; T, steps, m)
    end

    dt = T / steps
    for t in t+dt:dt:T
        @info "t = $t"
        evolve!(state, H.matrix, dt, m)
        write_norm(state, t, "$directory/norm.txt")
        write_ev(H, state, t, "$directory/H.txt")
        for operator in operators
            filename = "$directory/$(operator.name).txt"
            write_ev(operator, state, t, filename)
        end
        open("state.tmp", "w") do f
            serialize(f, (t, state))
        end
        mv("state.tmp", "states/$(parameters_str)__$m.bin"; force=true)
    end
end


function create_logger(filename)
    logger = LoggingExtras.FormatLogger() do io, args
        timestamp = Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS.sss")
        text = "$timestamp -- $(args.level) -- $(args.message)"
        println(stdout, text)
        open(filename, "a+") do f
            write(f, "$text\n")
        end
    end
    Logging.global_logger(logger)
end


function parse_(s::AbstractString)
    if occursin("/", s)
        num, den = split(s, "/")
        return parse_(num) / parse_(den)
    elseif occursin(".", s)
        return parse(Float64, s)
    else
        return parse(Int, s)
    end
end


function main()
    create_logger("julia.log")

    s = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table s begin
        "parameters"
            nargs = 7
        "--beta"
            arg_type = Float64
            default = 0.0
        "--abc"
            nargs = 4
        "--restore"
            action = :store_true
    end

    args = ArgParse.parse_args(s)

    j_0, c_0, t, j_max, T, steps, m = parse_.(args["parameters"])

    if args["abc"] == []
        model = "H_E"
        H = Operator("H_E", j_max)
        if args["beta"] != 0
            beta = args["beta"]
            model = "H_L/beta=$(beta)"
            H_E = H.matrix
            H_L = Operator("H_L", j_max).matrix
            H_matrix = 1 / beta^2 * H_E + (1 + beta^2) / beta^2 * H_L
            H = Operator{Float64}("H", j_max, H_matrix)
            H_E, H_L, H_matrix = nothing, nothing, nothing
        end
    else
        kind, a, b, c = args["abc"]
        _a, _b, _c = (x -> @sprintf("%.4g", x)).(parse_.([a, b, c]))
        model = "H_abc/$(kind)__$(_a)__$(_b)__$(_c)"
        rm("data/julia/$(j_max)/operators/H_abc.mtx"; force=true)
        H = Operator("H_abc", j_max, kind, a, b, c)
    end

    GC.gc()

    psi_0 = CoherentState(j_0, c_0, t, j_max)

    @time evolution(psi_0, H, T, steps, m;
                    model=model, restore=args["restore"])
end


main()
