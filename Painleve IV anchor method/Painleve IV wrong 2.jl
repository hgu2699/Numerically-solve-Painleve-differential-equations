# finite_n_fake_piv_locked_suite.jl
# Finite-n GUE largest-eigenvalue CDF:
#   - Fredholm side via Hermite kernel + Nyström on (s, s+L(n))
#   - "Painlevé side" via a FAKE σ-form Painlevé IV equation, locked to Fredholm.
#
# Fake σ-form:
#   (σ'')^2 = 4 (s σ' - σ + n)^2
#             - 4 (σ')^2 (σ' + 2n + 1)
#             + 3 (σ')^4
#
# This is NOT the correct GUE equation; it's for stress-testing only.

import Pkg
for p in ["OrdinaryDiffEq","FastGaussQuadrature","LinearAlgebra","Statistics","Plots"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using OrdinaryDiffEq, FastGaussQuadrature, LinearAlgebra, Statistics, Plots
gr()

# ---------------------------
# Orthonormal Hermite functions (weight e^{-x^2/2}):
# φ_0(x) = π^{-1/4} e^{-x^2/2}
# φ_1(x) = √2 x φ_0(x)
# φ_{k+1}(x) = √(2/(k+1)) x φ_k(x) - √(k/(k+1)) φ_{k-1}(x)
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
        return (ϕ1, sqrt(1.0)*x*ϕ1 - 1.0*ϕ0, sumsq)
    end
    ϕkm1, ϕk = ϕ0, ϕ1
    sumsq += ϕ1^2
    for k in 1:(n-2) # will end at k = n-2 producing ϕ_{n-1}
        α = sqrt(2.0/(k+1))
        β = sqrt(k/(k+1))
        ϕkp1 = α*x*ϕk - β*ϕkm1
        ϕkm1, ϕk = ϕk, ϕkp1
        sumsq += ϕk^2
    end
    # now ϕk = φ_{n-1}. One more step for φ_n:
    α = sqrt(2.0/n)
    β = sqrt((n-1)/n)
    ϕn = α*x*ϕk - β*ϕkm1
    return (ϕk, ϕn, sumsq)
end

# Christoffel–Darboux Hermite kernel with diagonal fallback
# K_n(s,t) = sqrt(n/2) * [φ_{n-1}(s) φ_n(t) - φ_n(s) φ_{n-1}(t)] / (s - t),   s≠t
# K_n(s,s) = ∑_{k=0}^{n-1} φ_k(s)^2
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
# Nyström with truncated semi-infinite interval:
# integrate on t ∈ [s, s+L], using Gauss–Legendre on [0,L], dt = (L/2) w
# ---------------------------
tail_len(n::Int) = n ≤ 50 ? 8.0 : (n ≤ 500 ? 10.0 : 12.0)

function F_n_fredholm(n::Int, s::Float64; N::Int=240, L::Float64=tail_len(n))
    # nodes on [0,L]
    z, w = gausslegendre(N)                  # (-1,1)
    u  = (z .+ 1.0) .* (L/2)                 # (0,L)
    dt = (L/2) .* w                          # weights for du
    t  = s .+ u                              # [s, s+L]
    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N
        tj = t[j]
        for i in 1:N
            A[i,j] = sq[i] * K_hermite(n, t[i], tj) * sq[j]
        end
    end
    λ = eigvals(Matrix(I - A))
    # Numerical guard
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

F_n_vec(n::Int, svals::AbstractVector{<:Real}; N::Int=240, L::Float64=tail_len(n)) =
    [F_n_fredholm(n, Float64(s); N=N, L=L) for s in svals]

# ---------------------------
# σ, σ', σ'' from log F via local LS fit (degree-4) on symmetric stencil
#
# log F(u) ≈ c1 + c2 u + c3 u^2 + c4 u^3 + c5 u^4
# ⇒ σ   = (logF)'(0)   = c2
#    σ'  = (logF)''(0)  = 2 c3
#    σ'' = (logF)'''(0) = 6 c4
# ---------------------------
function sigma_from_logF(spts::Vector{Float64}, Fvals::Vector{Float64})
    s0 = spts[cld(length(spts),2)]
    u  = spts .- s0
    y  = log.(Fvals)
    M  = hcat(u.^0, u, u.^2, u.^3, u.^4)
    c  = M \ y
    σ    = c[2]
    σp   = 2c[3]
    σpp0 = 6c[4]
    F0   = exp(c[1])
    return (; s0, σ, σp, σpp0, F0)
end

# ---------------------------
# Fake σ-form "Painlevé IV" and stepping with branch tracking
#
#   (σ'')^2 = 4 (s σ' - σ + n)^2
#             - 4 (σ')^2 (σ' + 2n + 1)
#             + 3 (σ')^4
# ---------------------------
@inline function piv_rhs(s::Float64, σp::Float64, σ::Float64, n::Int)
    term1 = 4.0 * (s*σp - σ + n)^2
    term2 = 4.0 * (σp^2) * (σp + 2.0*n + 1.0)
    term3 = 3.0 * (σp^4)
    return term1 - term2 + term3
end

@inline function choose_sigma_pp(prev_σpp::Float64, s::Float64,
                                 σ::Float64, σp::Float64, n::Int)
    rhs = piv_rhs(s, σp, σ, n)
    rhs ≤ 0 && return 0.0
    r = sqrt(rhs)

    # If we don't yet have a meaningful previous σ'', just take the + branch.
    if abs(prev_σpp) < 1e-10
        return r
    end

    c1 =  sign(prev_σpp) * r
    c2 = -sign(prev_σpp) * r
    return abs(c1 - prev_σpp) ≤ abs(c2 - prev_σpp) ? c1 : c2
end

function step_sigma!(σ::Float64, σp::Float64,
                     s_now::Float64, s_next::Float64,
                     n::Int, prev_σpp::Float64)
    h = s_next - s_now
    σpp = choose_sigma_pp(prev_σpp, s_now, σ, σp, n)
    σ_new  = σ  + h*σp + 0.5*h^2*σpp
    σp_mid = σp + 0.5*h*σpp
    σpp2   = choose_sigma_pp(σpp, 0.5*(s_now+s_next), σ_new, σp_mid, n)
    σp_new = σp + h*σpp2
    return σ_new, σp_new, σpp2
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
# One full run (fake PIV, robust Nyström)
# ---------------------------
function finite_n_fake_piv_locked(; n::Int=20, window_half::Float64=3.0,
                                   NquadFD::Int=260, npts::Int=1201, nanchors::Int=81)

    s_edge = sqrt(2.0*n)
    smin, smax = s_edge - window_half, s_edge + window_half
    @info "n=$n  edge≈$s_edge; domain [$smin,$smax]; Nquad=$NquadFD; anchors=$nanchors"

    # global s-grid (descending)
    sgrid = collect(range(smax, smin; length=npts))

    # anchors and local stencil
    anchors = collect(range(smax, smin; length=nanchors))
    h_anc   = (anchors[end]-anchors[1])/(length(anchors)-1)
    stencil = (-3:3) .* (0.5*h_anc)

    # Fredholm-based anchor data: σ, σ', σ'' from LS fit, plus F0
    anc = Vector{NamedTuple}(undef, length(anchors))
    for (j, s0) in pairs(anchors)
        spts = Float64[s0 + δ for δ in stencil]
        Fpts = F_n_vec(n, spts; N=NquadFD)   # robust truncated Nyström internally
        anc[j] = sigma_from_logF(spts, Fpts)
        (j % 5 == 0) && @info "  anchor $j/$(length(anchors)) at s0=$(round(s0,digits=3))"
    end

    # integrate σ between anchors with dynamic branch; reset at anchors
    σvals  = similar(sgrid)
    σpvals = similar(sgrid)

    # initial conditions from first anchor
    σvals[1]  = anc[1].σ
    σpvals[1] = anc[1].σp
    σpp_local = anc[1].σpp0
    k = 1

    for j in 1:(length(anchors)-1)
        sR = anchors[j+1]
        while k < length(sgrid) && sgrid[k] > sR + 1e-12
            σvals[k+1], σpvals[k+1], σpp_local =
                step_sigma!(σvals[k], σpvals[k], sgrid[k], sgrid[k+1], n, σpp_local)
            k += 1
        end
        # hard reset to Fredholm-based anchor (and its σ'')
        idx = findmin(abs.(sgrid .- sR))[2]
        σvals[idx]  = anc[j+1].σ
        σpvals[idx] = anc[j+1].σp
        σpp_local   = anc[j+1].σpp0
        k = idx
    end

    # reconstruct & compare
    F_piv = F_from_sigma_grid(sgrid, σvals, anc[1].s0, anc[1].F0)
    F_fd  = F_n_vec(n, sgrid; N=max(NquadFD,280))

    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-n Fredholm (Hermite kernel)",
                xlabel="s", ylabel="CDF",
                title="Finite-n GUE largest-eigenvalue CDF (n=$(n))")
    plot!(plt1, sgrid, F_piv, lw=2, ls=:dash, label="Fake PIV σ-form (locked)")
    savefig(plt1, "finite_n_fake_piv_locked_vs_fd_n$(n).png"); display(plt1)

    plt2 = plot(sgrid, abs.(F_fd .- F_piv), lw=2, label="|Δ|",
                xlabel="s", ylabel="absolute error",
                title="Absolute difference: fake PIV (locked) vs Fredholm (n=$(n))")
    savefig(plt2, "finite_n_fake_piv_locked_absdiff_n$(n).png"); display(plt2)

    @info "n=$n: max |Δ| = $(maximum(abs.(F_fd .- F_piv)))"
    return (sgrid=sgrid, F_fd=F_fd, F_piv=F_piv, plt1=plt1, plt2=plt2)
end

# ---------------------------
# Run the suite
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
        finite_n_fake_piv_locked(n=c.n, NquadFD=c.N,
                                 nanchors=c.anchors, window_half=c.window)
    end
    @info "Saved all figures in $(pwd())"
end

# Always run so you see plots/files even via include()
run_suite()
