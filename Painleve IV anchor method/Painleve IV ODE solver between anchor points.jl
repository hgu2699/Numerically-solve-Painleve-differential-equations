# finite_n_piv_locked_correct_sigma_ode.jl
#
# Painlevé IV σ-form with branch-locked *ODE solver* between anchors.
#
#   (σ'')^2 = 4 (s σ' - σ)^2 - 4 (σ')^2 (σ' + 2n)
#
# Anchors (σ, σ', σ'') are extracted from log F_n(s) via local LS polynomial
# fits; between anchors we integrate the σ-equation as a 1st-order system
#
#   σ'  = σ_p
#   σ_p' = sign(σ''(anchor)) * sqrt(RHS(s,σ,σ_p))
#
# with fixed branch sign on each interval.

import Pkg
for p in ["OrdinaryDiffEq","FastGaussQuadrature","LinearAlgebra","Statistics","Plots"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using OrdinaryDiffEq
using FastGaussQuadrature
using LinearAlgebra
using Statistics
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
# σ, σ', σ'' from local LS fit of log F (degree-4 poly)
# ---------------------------
function sigma_from_logF(spts::Vector{Float64}, Fvals::Vector{Float64})
    s0 = spts[cld(length(spts),2)]
    u  = spts .- s0
    y  = log.(Fvals)
    M  = hcat(u.^0, u, u.^2, u.^3, u.^4)
    c  = M \ y
    σ    = c[2]
    σp   = 2.0 * c[3]
    σpp0 = 6.0 * c[4]
    F0   = exp(c[1])
    return (; s0, σ, σp, σpp0, F0)
end

# ---------------------------
# Correct PIV σ-form
#
#   (σ'')^2 = 4 (s σ' - σ)^2 - 4 (σ')^2 (σ' + 2n)
# ---------------------------
@inline function piv_rhs(s::Float64, σp::Float64, σ::Float64, n::Int)
    return 4.0*(s*σp - σ)^2 - 4.0*(σp^2)*(σp + 2.0*n)
end

# σ'' with fixed branch sign on an interval
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
# Reconstruct CDF from σ via trapezoid rule
# ---------------------------
function F_from_sigma_grid(sgrid::Vector{Float64}, σvals::Vector{Float64},
                           s0::Float64, F0::Float64)
    n = length(sgrid)
    logF = zeros(n)
    k0   = findmin(abs.(sgrid .- s0))[2]
    logF[k0] = log(F0)

    # integrate upwards in index (towards larger s)
    for k in (k0-1):-1:1
        dx = sgrid[k+1] - sgrid[k]
        logF[k] = logF[k+1] + 0.5*(σvals[k] + σvals[k+1])*dx
    end

    # integrate downwards in index (towards smaller s)
    for k in (k0+1):n
        dx = sgrid[k] - sgrid[k-1]
        logF[k] = logF[k-1] + 0.5*(σvals[k] + σvals[k-1])*dx
    end
    return exp.(logF)
end

# ---------------------------
# One full run: PIV with ODE between anchors
# ---------------------------
function finite_n_piv_locked_correct_ode(; n::Int=20, window_half::Float64=3.0,
                                         NquadFD::Int=260, npts::Int=1201,
                                         nanchors::Int=81)

    s_edge = sqrt(2.0*n)
    smin, smax = s_edge - window_half, s_edge + window_half
    @info "CORRECT σ-form (ODE): n=$n  edge≈$s_edge; domain [$smin,$smax]; Nquad=$NquadFD; anchors=$nanchors"

    # global s-grid (right to left)
    sgrid = collect(range(smax, smin; length=npts))

    # anchors (also right to left)
    anchors = collect(range(smax, smin; length=nanchors))
    h_anc   = (anchors[end] - anchors[1]) / (length(anchors)-1)
    stencil = (-3:3) .* (0.5*h_anc)

    # compute anchor data from Fredholm side
    anc = Vector{NamedTuple}(undef, length(anchors))
    for (j, s0) in pairs(anchors)
        spts = Float64[s0 + δ for δ in stencil]
        Fpts = F_n_vec(n, spts; N=NquadFD)
        anc[j] = sigma_from_logF(spts, Fpts)
        (j % 5 == 0) && @info "  anchor $j/$(length(anchors)) at s0=$(round(s0,digits=3))"
    end

    # storage for σ and σ' on the global grid
    σvals  = fill(NaN, length(sgrid))
    σpvals = fill(NaN, length(sgrid))

    # Make sure the first anchor is put on the grid explicitly
    idx1 = findmin(abs.(sgrid .- anchors[1]))[2]
    σvals[idx1]  = anc[1].σ
    σpvals[idx1] = anc[1].σp

    # Solve between consecutive anchors with ODE solver, re-anchoring at each step.
    for j in 1:(length(anchors)-1)
        sL = anchors[j]     # right endpoint of the interval
        sR = anchors[j+1]   # left endpoint of the interval (sR < sL)

        σ0    = anc[j].σ
        σp0   = anc[j].σp
        σpp0  = anc[j].σpp0
        sgnσpp = abs(σpp0) < 1e-10 ? 1.0 : sign(σpp0)

        u0 = [σ0, σp0]
        p  = (n, sgnσpp)

        # s-grid points lying in [sR, sL] (inclusive), note sgrid is descending
        idx_range = findall(s -> (s ≤ sL + 1e-10) && (s ≥ sR - 1e-10), sgrid)
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

        # Re-anchor at the left endpoint using the LS fit for anchor j+1
        idxR = findmin(abs.(sgrid .- sR))[2]
        σvals[idxR]  = anc[j+1].σ
        σpvals[idxR] = anc[j+1].σp
    end

    # Build CDF from σ
    F_piv = F_from_sigma_grid(sgrid, σvals, anc[1].s0, anc[1].F0)
    F_fd  = F_n_vec(n, sgrid; N=max(NquadFD, 280))

    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-n Fredholm (Hermite kernel)",
                xlabel="s", ylabel="CDF",
                title="Finite-n GUE CDF (CORRECT σ-form, ODE, n=$(n))")
    plot!(plt1, sgrid, F_piv, lw=2, ls=:dash,
          label="PIV σ-form (correct, ODE between anchors)")
    savefig(plt1, "finite_n_piv_correct_ode_vs_fd_n$(n).png"); display(plt1)

    plt2 = plot(sgrid, abs.(F_fd .- F_piv), lw=2, label="|Δ|",
                xlabel="s", ylabel="absolute error",
                title="Abs diff: correct PIV (ODE) vs Fredholm (n=$(n))")
    savefig(plt2, "finite_n_piv_correct_ode_absdiff_n$(n).png"); display(plt2)

    @info "CORRECT σ-form (ODE): n=$n: max |Δ| = $(maximum(abs.(F_fd .- F_piv)))"
    return (sgrid=sgrid, F_fd=F_fd, F_piv=F_piv, plt1=plt1, plt2=plt2)
end

# ---------------------------
# Run the suite (CORRECT σ-form, ODE version)
# ---------------------------
function run_suite()
    configs = [
        (n=5,    N=220, anchors=61,  window=3.5),
        (n=10,   N=240, anchors=81,  window=3.5),
        (n=20,   N=260, anchors=91,  window=3.2),
        (n=100,  N=280, anchors=101, window=3.0),
        (n=500,  N=300, anchors=121, window=3.0),
    ]
    for c in configs
        finite_n_piv_locked_correct_ode(n=c.n, NquadFD=c.N,
                                        nanchors=c.anchors,
                                        window_half=c.window)
    end
    @info "CORRECT σ-form (ODE): saved all figures in $(pwd())"
end

run_suite()
