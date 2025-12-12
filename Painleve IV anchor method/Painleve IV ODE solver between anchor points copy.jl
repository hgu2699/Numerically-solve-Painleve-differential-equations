# finite_n_piv_fd_ode_adaptive.jl
#
# Step 1: run FD+ODE method with uniform anchors (no plots) to get error profile.
# Step 2: find the 3 largest error locations in s.
# Step 3: add these s-points to the anchors and rerun FD+ODE.
# Step 4: plot F_n (Fredholm) vs PIV and the new abs error.
#
# Done for n = 5, 20, 100, 500.

import Pkg
for p in ["OrdinaryDiffEq","FastGaussQuadrature","LinearAlgebra","Plots"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using OrdinaryDiffEq
using FastGaussQuadrature
using LinearAlgebra
using Plots
gr()

# ---------------------------
# Orthonormal Hermite functions (weight e^{-x^2/2})
# ---------------------------
function hermite_phi_chain(n::Int, x::Float64)
    # returns (φ_{n-1}(x), φ_n(x), sum_{k=0}^{n-1} φ_k(x)^2)
    ϕ0 = pi^(-0.25) * exp(-0.5*x*x)

    if n == 1
        return (ϕ0, sqrt(2.0)*x*ϕ0, ϕ0^2)
    end

    ϕ1 = sqrt(2.0)*x*ϕ0
    sumsq = ϕ0^2

    if n == 2
        sumsq += ϕ1^2
        ϕ2 = sqrt(1.0)*x*ϕ1 - 1.0*ϕ0
        return (ϕ1, ϕ2, sumsq)
    end

    ϕkm1, ϕk = ϕ0, ϕ1
    sumsq += ϕ1^2
    for k in 1:(n-2)
        α = sqrt(2.0/(k+1))
        β = sqrt(k/(k+1))
        ϕkp1 = α*x*ϕk - β*ϕkm1
        ϕkm1, ϕk = ϕk, ϕkp1
        sumsq += ϕk^2
    end
    α = sqrt(2.0/n)
    β = sqrt((n-1)/n)
    ϕn = α*x*ϕk - β*ϕkm1
    return (ϕk, ϕn, sumsq)
end

# Christoffel–Darboux Hermite kernel with diagonal fallback
function K_hermite(n::Int, s::Float64, t::Float64)
    if s == t
        _, _, sumsq = hermite_phi_chain(n, s)
        return sumsq
    end
    φnm1_s, φn_s, _ = hermite_phi_chain(n, s)
    φnm1_t, φn_t, _ = hermite_phi_chain(n, t)
    return sqrt(n/2) * (φnm1_s*φn_t - φn_s*φnm1_t) / (s - t)
end

# ---------------------------
# Fredholm determinant via Nyström on [s, s+L(n)]
# ---------------------------
tail_len(n::Int) = n ≤ 50 ? 8.0 : (n ≤ 500 ? 10.0 : 12.0)

function F_n_fredholm(n::Int, s::Float64; N::Int=240, L::Float64=tail_len(n))
    z, w = gausslegendre(N)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u
    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N
        tj = t[j]
        for i in 1:N
            A[i,j] = sq[i] * K_hermite(n, t[i], tj) * sq[j]
        end
    end
    λ = eigvals(Matrix(I - A))
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

F_n_vec(n::Int, svals::AbstractVector{<:Real}; N::Int=240, L::Float64=tail_len(n)) =
    [F_n_fredholm(n, Float64(s); N=N, L=L) for s in svals]

# ---------------------------
# Finite-difference derivatives of log F at an anchor
# y(s) = log F_n(s)
#   σ    = y'(s0)
#   σ'   = y''(s0)
#   σ''  = y'''(s0)
# 7-point symmetric, O(h^6) formulas
# ---------------------------
function sigma_from_logF_fd(n::Int, s0::Float64; h::Float64, NquadFD::Int)
    ks   = collect(-3:3)
    spts = s0 .+ h .* Float64.(ks)
    Fpts = F_n_vec(n, spts; N=NquadFD)
    y    = log.(Fpts)

    c1 = [-1.0/60,  3.0/20, -3.0/4, 0.0,  3.0/4, -3.0/20,  1.0/60]
    c2 = [ 1.0/90, -3.0/20,  3.0/2, -49.0/18, 3.0/2, -3.0/20, 1.0/90]
    c3 = [ 1.0/8,  -1.0,     13.0/8, 0.0, -13.0/8, 1.0,     -1.0/8]

    σ   = sum(c1 .* y) / h
    σp  = sum(c2 .* y) / (h*h)
    σpp = sum(c3 .* y) / (h*h*h)

    F0  = Fpts[4]
    return (; s0, σ, σp, σpp0 = σpp, F0)
end

# ---------------------------
# Correct PIV σ-form
#   (σ'')^2 = 4 (s σ' - σ)^2 - 4 (σ')^2 (σ' + 2n)
# ---------------------------
@inline function piv_rhs(s::Float64, σp::Float64, σ::Float64, n::Int)
    return 4.0*(s*σp - σ)^2 - 4.0*(σp^2)*(σp + 2.0*n)
end

@inline function sigma_pp_signed(s::Float64, σ::Float64, σp::Float64,
                                 n::Int, sgnσpp::Float64)
    rhs = piv_rhs(s, σp, σ, n)
    rhs <= 0.0 && return 0.0
    return sgnσpp * sqrt(rhs)
end

# 1st-order ODE system for (σ, σ')
function sigma_ode!(du, u, p, s)
    σ   = u[1]
    σp  = u[2]
    n, sgnσpp = p
    du[1] = σp
    du[2] = sigma_pp_signed(s, σ, σp, n, sgnσpp)
end

# ---------------------------
# Reconstruct F from σ via trapezoid rule on a global s-grid
# ---------------------------
function F_from_sigma_grid(sgrid::Vector{Float64}, σvals::Vector{Float64},
                           s0::Float64, F0::Float64)
    n = length(sgrid)
    logF = zeros(n)
    k0   = findmin(abs.(sgrid .- s0))[2]
    logF[k0] = log(F0)

    for k in (k0-1):-1:1
        dx = sgrid[k+1] - sgrid[k]
        logF[k] = logF[k+1] + 0.5*(σvals[k] + σvals[k+1])*dx
    end
    for k in (k0+1):n
        dx = sgrid[k] - sgrid[k-1]
        logF[k] = logF[k-1] + 0.5*(σvals[k] + σvals[k-1])*dx
    end
    return exp.(logF)
end

# ---------------------------
# Core FD+ODE solver on a given s-grid and anchor set
# (this is your "old method", just factored out)
# ---------------------------
function solve_sigma_piv_on_grid_fd_ode(n::Int,
                                        sgrid::Vector{Float64},
                                        anchors_in::Vector{Float64};
                                        NquadFD::Int)

    anchors = sort(collect(anchors_in); rev=true)  # smax → smin

    # choose FD step from average anchor spacing
    h_anc = (anchors[1] - anchors[end]) / (length(anchors)-1)
    h_fd  = 0.5 * h_anc

    # anchor data via Nyström + finite differences
    anc = Vector{NamedTuple}(undef, length(anchors))
    for (j, s0) in pairs(anchors)
        anc[j] = sigma_from_logF_fd(n, s0; h=h_fd, NquadFD=NquadFD)
    end

    σvals  = fill(NaN, length(sgrid))
    σpvals = fill(NaN, length(sgrid))

    # put first anchor on grid
    idx1 = findmin(abs.(sgrid .- anchors[1]))[2]
    σvals[idx1]  = anc[1].σ
    σpvals[idx1] = anc[1].σp

    # integrate between successive anchors using ODE solver
    for j in 1:(length(anchors)-1)
        sL = anchors[j]
        sR = anchors[j+1]      # sR < sL

        σ0     = anc[j].σ
        σp0    = anc[j].σp
        σpp0   = anc[j].σpp0
        sgnσpp = abs(σpp0) < 1e-10 ? 1.0 : sign(σpp0)

        u0 = [σ0, σp0]
        p  = (n, sgnσpp)

        idx_range = findall(s -> (s ≤ sL + 1e-12) && (s ≥ sR - 1e-12), sgrid)
        isempty(idx_range) && continue

        save_times = sgrid[idx_range]
        prob = ODEProblem(sigma_ode!, u0, (sL, sR), p)
        sol  = solve(prob, Vern7(), abstol=1e-10, reltol=1e-10,
                     saveat=save_times)

        @assert length(sol.u) == length(idx_range)

        for (k_local, k_grid) in enumerate(idx_range)
            σvals[k_grid]  = sol.u[k_local][1]
            σpvals[k_grid] = sol.u[k_local][2]
        end

        # re-anchor at sR
        idxR = findmin(abs.(sgrid .- sR))[2]
        σvals[idxR]  = anc[j+1].σ
        σpvals[idxR] = anc[j+1].σp
    end

    return σvals, σpvals, anc, anchors
end

# ---------------------------
# One uniform run (old method), NO plotting.
# Returns sgrid, F_fd, F_piv, error, anchors, anc data.
# ---------------------------
function run_uniform_fd_ode_once(; n::Int,
                                  window_half::Float64,
                                  NquadFD::Int,
                                  npts::Int,
                                  nanchors::Int)

    s_edge = sqrt(2.0*n)
    smin, smax = s_edge - window_half, s_edge + window_half
    sgrid = collect(range(smax, smin; length=npts))

    anchors_uniform = collect(range(smax, smin; length=nanchors))

    σvals, σpvals, anc, anchors_uniform =
        solve_sigma_piv_on_grid_fd_ode(n, sgrid, anchors_uniform; NquadFD=NquadFD)

    F_piv = F_from_sigma_grid(sgrid, σvals, anc[1].s0, anc[1].F0)
    F_fd  = F_n_vec(n, sgrid; N=max(NquadFD, 280))

    err = abs.(F_fd .- F_piv)

    return (sgrid=sgrid, F_fd=F_fd, F_piv=F_piv, err=err,
            anchors=anchors_uniform, anc=anc,
            smin=smin, smax=smax)
end

# ---------------------------
# Pick top-K error locations (optionally enforcing separation)
# ---------------------------
function top_error_points(sgrid::Vector{Float64}, err::Vector{Float64};
                          K::Int=3, min_sep::Float64=0.05)
    idx_sorted = sortperm(err, rev=true)
    chosen = Int[]
    for idx in idx_sorted
        # enforce minimal distance in s so we don't pick almost-identical points
        if all(abs(sgrid[idx] - sgrid[j]) > min_sep for j in chosen)
            push!(chosen, idx)
            length(chosen) == K && break
        end
    end
    return sgrid[chosen], chosen
end

# ---------------------------
# Adaptive run:
#   1) uniform old method, find top-3 error points
#   2) add them to anchors, rerun FD+ODE
#   3) plot F and abs error
# ---------------------------
function run_adaptive_fd_ode(; n::Int,
                              window_half::Float64,
                              NquadFD::Int,
                              npts::Int,
                              nanchors::Int,
                              K::Int=3)

    base = run_uniform_fd_ode_once(; n=n, window_half=window_half,
                                    NquadFD=NquadFD, npts=npts,
                                    nanchors=nanchors)

    sgrid = base.sgrid
    F_fd  = base.F_fd
    err   = base.err
    anchors_uniform = base.anchors

    s_hot, _ = top_error_points(sgrid, err; K=K, min_sep=0.05)
    @info "n=$n: base max |Δ| = $(maximum(err)); hot spots = $(s_hot)"

    anchors_adapt = unique(sort(vcat(anchors_uniform, s_hot); rev=true))

    σvals_adapt, σpvals_adapt, anc_adapt, anchors_adapt =
        solve_sigma_piv_on_grid_fd_ode(n, sgrid, anchors_adapt; NquadFD=NquadFD)

    F_piv_adapt = F_from_sigma_grid(sgrid, σvals_adapt,
                                    anc_adapt[1].s0, anc_adapt[1].F0)
    err_adapt = abs.(F_fd .- F_piv_adapt)

    @info "n=$n: adaptive max |Δ| = $(maximum(err_adapt))"

    # Plots for the adaptive run
    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-n Fredholm (Hermite)",
                xlabel="s", ylabel="CDF",
                title="Finite-n GUE CDF (FD+ODE adaptive anchors, n=$(n))")
    plot!(plt1, sgrid, F_piv_adapt, lw=2, ls=:dash,
          label="PIV σ (adaptive anchors)")
    savefig(plt1, "finite_n_piv_fd_ode_adapt_vs_fd_n$(n).png"); display(plt1)

    plt2 = plot(sgrid, err_adapt, lw=2, label="|Δ| adaptive",
                xlabel="s", ylabel="absolute error",
                title="Abs diff: adaptive anchors vs Fredholm (n=$(n))")
    savefig(plt2, "finite_n_piv_fd_ode_adapt_absdiff_n$(n).png"); display(plt2)

    return (sgrid=sgrid, F_fd=F_fd, F_piv_adapt=F_piv_adapt,
            err_base=err, err_adapt=err_adapt,
            anchors_uniform=anchors_uniform,
            anchors_adapt=anchors_adapt,
            hot_spots=s_hot)
end

# ---------------------------
# Suite driver for n = 5,20,100,500
# ---------------------------
function run_suite_adaptive()
    configs = [
        (n=5,   N=220, anchors=61,  window=3.5),
        (n=20,  N=260, anchors=91,  window=3.2),
        (n=100, N=280, anchors=101, window=3.0),
        (n=500, N=300, anchors=121, window=3.0),
    ]
    for c in configs
        run_adaptive_fd_ode(n=c.n, window_half=c.window,
                            NquadFD=c.N, npts=1201,
                            nanchors=c.anchors, K=3)
    end
    @info "Adaptive FD+ODE suite finished; figures saved in $(pwd())."
end

run_suite_adaptive()
