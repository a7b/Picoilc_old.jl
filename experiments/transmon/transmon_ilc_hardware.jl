# using Pkg
# ENV["PYTHON"] = Sys.which("python")
# Pkg.build("PyCall")
using Pico
using PyCall
using Statistics
using JLD2
using HDF5

include(joinpath(@__DIR__, "measurement.jl"))

file = h5open("Z:\\Pico.jl\\experiments\\transmon\\pi_y_test_gauss.h5", "r")
uks = read(file, "uks")
total_time = read(file, "total_time")
us = Vector{Float64}[]
steps = size(uks,1)
for i = 1:steps
    push!(us, uks[i, :, 1])
end
# function gaussian_controls()
#     #0.9545
#     MAX_DRIVE = A_MAX/3
#     time_sigma = (π/2)/0.9545/(MAX_DRIVE * sqrt(2π))
#     times = LinRange(-2*time_sigma, 2*time_sigma, 100)
#     time = time_sigma*4
#     dt = times[2] - times[1]
#     contx = MAX_DRIVE * Base.exp.(-(times).^2/(2*time_sigma^2))
#     acontrols = [[contx[k], 0.0] for k=1:length(contx)]
#     #print(acontrols)
#     append!(acontrols, [[0.0,0.0]])
#     #print(acontrols)
#     return (acontrols, time, dt)
# end
#factor to boost analytic rollout
us = 0.97 .* us
display(us)
print(steps)

dt = total_time/steps
@pyinclude "experiments/transmon/run_experiment_optimize_loop.py"

# #need to change this to take in the jld2 file from the transmon solve
# # @Aaron did you use the operator basis to generate the pulse?

# # data_dir = "data_tracked/multimode/good_solutions"

# # data_name = "g0_to_g1_T_101_dt_4.0_Q_500.0_R_0.1_u_bound_1.0e-5"

# # data_path = joinpath(data_dir, data_name * ".jld2")



data_path = "experiments/transmon/5-4-2023-transmon_no_int_a_iter_3000_time_40.0ns_pinq_true_00000.jld2"

run1_path = "log/transmon/hardware/ILC/taus_200_300_400_00001.jld2"
run1_path = "Z:/Pico.jl/log/transmon/hardware/ILC/taus_200_300_400real_log_sample_12k_00000.jld2"
#run1_path = "Z:/Pico.jl/data/ilc/hardware125k_3meas_200_300_400_iter_6.jld2"
data = load_data(data_path)

# @load run1_path traj ys Us

# traj = prob.Ẑ

# println(data.trajectory)
# println(data.traject

dts = [dt for t = 1:steps]
xs_r = rollout(data.system, us, dts)
xs = [xs_r[t][1:2data.system.isodim] for t = 1:steps]
times = [i * dt for i = 0:steps-1]
display(times)
#display(us)
Ẑ = Trajectory(
    xs,
    us,
    times,
    steps,
    dt
)
# # xs = [
# #     data.trajectory.states[t][1:2data.system.isodim]
# #         for t = 1:data.trajectory.T
# # ]

# # us = [
# #     data.trajectory.states[t][
# #         (data.system.n_wfn_states +
# #         data.system.∫a * data.system.ncontrols) .+
# #         (1:data.system.ncontrols)
# #     ] for t = 1:data.trajectory.T
# # ]
# # Ẑ = Trajectory(
# #     xs,
# #     us,
# #     data.trajectory.times,
# #     data.trajectory.T,
# #     data.trajectory.Δt
# # )

# # # T = data.trajectory.T


# function g_pauli(x)
#     y = []
#     for i = 1:2
#         ψ_i = x[slice(i, data.system.isodim)]
#         append!(y, (-meas_x_iso(ψ_i) + 1) / 2)
#         append!(y, (-meas_y_iso(ψ_i) + 1) / 2)
#         append!(y, (-meas_z_iso(ψ_i) + 1) / 2)
#     end
#     return convert(typeof(x), y)
# end

function g_fidelity(x)
    y = []
    ψ1 = iso_to_ket(x[slice(1, data.system.isodim)])
    ψ2 = iso_to_ket(x[slice(2, data.system.isodim)])
    U = hcat(ψ1, ψ2)
    U_goal = [0 -im; -im 0]
    append!(y, 1/2*abs(tr(U' * U_goal)))
    return convert(typeof(x), y)
end
# # y_goal = g_pauli(data.trajectory.states[end])
y_goal = g_fidelity(xs[end])
println(y_goal)
ydim = length(y_goal)
τs = [steps]

function g_hardware(
    us::Vector{Vector{Float64}},
    times::AbstractVector{Float64},
    τs::AbstractVector{Int},
    samples::Int
)::MeasurementData
    # display(us)
    ys = py"take_controls_and_measure"(times, us, τs, samples, rb=true) |> transpose
    #ys = zeros(6,3)
    display(ys)
    println()
    println("y_goal")
    display(y_goal)
    println(typeof(ys))
    ys = collect(eachcol(ys))
    return MeasurementData(ys, τs, ydim)
end





# # # function build_cov_matrix(N=20)
# # #     us = Ẑ.actions
# # #     ys = []
# # #     for _ = 1:N
# # #         y = experiment(us).ys[end]
# # #         push!(ys, y)
# # #     end
# # #     Y = hcat(ys...)
# # #     return cov(Y; dims=2)
# # # end

# # # Σ = build_cov_matrix(15)

# # # @info "cov matrix"
# # # display(Σ)

max_iter = 3
max_backtrack_iter = 4
fps = 2
α = 0.5
β = 0.01
R = 1.0e-1
Qy = 1.0e1
Qf = 1.0e4

τs = [steps]

for samples in [
 1000,
]
    print("Running with samples = $samples")
    save_dir = "hardware_ILC"
    save_name = "gauss_rb_1000_quick_final" * join(τs, "_")

    experiment = HardwareExperiment(
        (U, ts, taus) -> g_hardware(U, ts, taus, samples),
        g_fidelity,
        τs,
        times,
        ydim
    )

    prob = ILCProblem(
        data.system,
        Ẑ,
        experiment;
        max_iter=max_iter,
        QP_verbose=false,
        correction_term=true,
        norm_p=1,
        R=R,
        static_QP=false,
        Qy=Qy,
        Qf=Qf,
        use_system_goal=true,
        α=α,
        β=β,
        # Σ=Σ,
        max_backtrack_iter=max_backtrack_iter,
        exp_name="gauss_rb_1000_quick_final"
    )



    log_dir = "log/transmon/hardware/ILC"
    log_path = generate_file_path("jld2", save_name, log_dir)
    real_log_path = generate_file_path("jld2", save_name*"real_log_sample", log_dir)

    outfile = generate_file_path("txt", "output_samples_$(samples)", @__DIR__)

    io = open(outfile, "w")

    solve!(prob; io=io)

    close(io)

    @save log_path prob
    #change to local
    local traj = prob.Ẑ
    local ys = prob.Ȳs
    local Us = prob.Us
    @save real_log_path traj ys Us

    plot_dir = "plots/transmon/hardware/ILC/"
    plot_path = generate_file_path("gif", save_name, plot_dir)
    #would have to check if this plotting function still works
    animate_ILC(prob, plot_path; fps=fps)
end
