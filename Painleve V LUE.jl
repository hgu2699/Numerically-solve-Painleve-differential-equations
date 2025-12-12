# finite_n_pv_lue_locked_suite.jl
# Verifies finite-N LUE largest-eigenvalue CDF using:
# (i) Fredholm determinant with Laguerre (CD + diagonal fallback) kernel and truncated Nyström on (s, ∞)
# (ii) σ–Painlevé V with anchor-locked branch tracking (LS fit on log F)
#
# Output: lue_pv_locked_vs_fd_N<N>_a<a>.png and lue_pv_locked_absdiff_N<N>_a<a>.png

import Pkg
for p in ["FastGaussQuadrature","LinearAlgebra","Statistics","Plots","SpecialFunctions"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature, LinearAlgebra, Statistics, Plots, SpecialFunctions
gr()

# ---------------------------
# Orthonormal Laguerre polynomials for weight w(x)=x^α e^{-x} on (0,∞)
# Recurrence for orthonormal p_n:
#   x p_n = a_{n+1} p_{n+1} + b_n p_n + a_n p_{n-1},  with
#   a_n = √(n (n+α)),  b_n = 2n + α + 1,  (a_0 := 0).
# Orthonormal functions: φ_n(x) = p_n(x) * sqrt(w(x)),  w(x)=x^α e^{-x}.
# ---------------------------
@inline function laguerre_phi_chain(N::Int, α::Float64, x::Float64)
    # returns (φ_{N-1}(x), φ_N(x), sum_{k=0}^{N-1} φ_k(x)^2)
    # Handle x=0 safely
    if x < 0.0
        error("Laguerre support is x∈[0,∞). Got x=$x")
    end
    # normalization for p_0: ∫ p_0^2 w = 1 => p_0 = 1/√Γ(α+1)
    p0 = 1.0 / sqrt(gamma(α + 1.0))
    if N == 1
        sqrtw = (x > 0 ? x^(0.5*α) : (α == 0 ? 1.0 : 0.0)) * exp(-0.5*x)  # x^(α/2) e^{-x/2}
        φ0 = p0 * sqrtw
        # from recurrence for n=0: x p0 = a1 p1 + b0 p0 => p1 = (x - b0)p0/a1
        a1 = sqrt(1.0 * (1.0 + α))
        b0 = 2.0*0.0 + α + 1.0
        p1 = (x - b0) * p0 / a1
        φ1 = p1 * sqrtw
        return (φ0, φ1, φ0^2)
    end
    # build p_k up to k=N with forward recurrence
    a = x -> x  # just placeholder to avoid name clash
    sqrtw = (x > 0 ? x^(0.5*α) : (α == 0 ? 1.0 : 0.0)) * exp(-0.5*x)
    pkm1 = 0.0
    pk   = p0
    sum_p2 = pk^2  # accumulate p_k^2; multiply by w at the end
    # first step: get p1
    a1 = sqrt(1.0 * (1.0 + α))
    b0 = α + 1.0
    p1 = (x - b0) * pk / a1
    pkm1, pk = pk, p1
    sum_p2 += pk^2
    if N == 2
        φnm1 = pkm1 * sqrtw  # p0
        φn   = pk   * sqrtw  # p1
        return (φn, ( (x - (2.0+α+1.0)) * pk - sqrt(1.0*(1.0+α)) * pkm1 ) / sqrt(2.0*(2.0+α)) * sqrtw, sum_p2 * (sqrtw^2))
    end
    for n in 1:(N-2) # will end with pk = p_{N-1}
        an1 = sqrt((n+1.0)*((n+1.0)+α))   # a_{n+1}
        bn  = 2.0*n + α + 1.0             # b_n
        an  = sqrt(n*(n+α))               # a_n
        pkp1 = (x*pk - bn*pk - an*pkm1) / an1
        pkm1, pk = pk, pkp1
        sum_p2 += pk^2
    end
    # pk now is p_{N-1}; compute p_N one more step
    aN  = sqrt(N*(N+α))          # a_N
    bNm1 = 2.0*(N-1) + α + 1.0   # b_{N-1}
    pN = (x*pk - bNm1*pk - sqrt((N-1.0)*(N-1.0+α))*pkm1) / aN
    φnm1 = pk * sqrtw
    φn   = pN * sqrtw
    return (φnm1, φn, sum_p2 * (sqrtw^2))
end

# Laguerre (orthonormal) CD kernel with diagonal fallback
# K_N(x,y) = a_N [φ_N(x) φ_{N-1}(y) - φ_{N-1}(x) φ_N(y)]/(x - y),   x≠y
# K_N(x,x) = ∑_{k=0}^{N-1} φ_k(x)^2
@inline function K_laguerre(N::Int, α::Float64, x::Float64, y::Float64)
    if x == y
        _, _, sumsq = laguerre_phi_chain(N, α, x)
        return sumsq
    end
    φNm1_x, φN_x, _ = laguerre_phi_chain(N, α, x)
    φNm1_y, φN_y, _ = laguerre_phi_chain(N, α, y)
    aN = sqrt(N*(N+α))
    return aN * (φN_x*φNm1_y - φNm1_x*φN_y) / (x - y)
end

# ---------------------------
# Nyström on (s, ∞): truncate to (s, s+L), Gauss–Legendre on [0,L]
# ---------------------------
tail_len_lue(N::Int, α::Float64) = 10.0 * max(1.0, N^(1/3))  # heuristic; safe but not huge

function F_N_fredholm_LUE(N::Int, α::Float64, s::Float64; Nquad::Int=240, L::Float64=tail_len_lue(N,α))
    # nodes on [0,L]
    z, w = gausslegendre(Nquad)            # (-1,1)
    u  = (z .+ 1.0) .* (L/2)               # (0,L)
    dt = (L/2) .* w                        # weights for du
    t  = s .+ u                             # [s, s+L]
    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, Nquad, Nquad)
    @inbounds for j in 1:Nquad
        tj = t[j]
        for i in 1:Nquad
            A[i,j] = sq[i] * K_laguerre(N, α, t[i], tj) * sq[j]
        end
    end
    λ = eigvals(Matrix(I - A))
    λ = clamp.(real.(λ), eps(), 1.0)   # numerical guard
    return exp(sum(log, λ))
end

F_N_vec_LUE(N::Int, α::Float64, svals::AbstractVector{<:Real}; Nquad::Int=240, L::Float64=tail_len_lue(N,α)) =
    [F_N_fredholm_LUE(N, α, Float64(s); Nquad=Nquad, L=L) for s in svals]

# ---------------------------
# σ, σ' from local LS fit of log F on symmetric stencil (deg-4)
# ---------------------------
function sigma_from_logF(spts::Vector{Float64}, Fvals::Vector{Float64})
    s0 = spts[cld(length(spts),2)]
    u  = spts .- s0
    y  = log.(Fvals)
    M  = hcat(u.^0, u, u.^2, u.^3, u.^4)
    c  = M \ y
    σ   = c[2]
    σp  = 2c[3]
    F0  = exp(c[1])
    return (; s0, σ, σp, F0)
end

# ---------------------------
# σ–Painlevé V (Jimbo–Miwa–Okamoto σ-form)
# (t σ'')^2 = [ σ - t σ' + 2(σ')^2 + S σ' ]^2 - 4 ∏_{j=0}^3 (σ' + ν_j),
# where S = ν0+ν1+ν2+ν3
# We'll use parameters ν = (0, 0, N+α, N) for the standard gap case.
# ---------------------------
@inline function sigma_pv_rhs_sq(t::Float64, σp::Float64, σ::Float64, ν::NTuple{4,Float64})
    S = ν[1] + ν[2] + ν[3] + ν[4]
    A = σ - t*σp + 2.0*σp^2 + S*σp
    B = (σp + ν[1]) * (σp + ν[2]) * (σp + ν[3]) * (σp + ν[4])
    return A*A - 4.0*B   # equals (t σ'')^2
end

@inline function choose_sigma_pp_pv(prev_σpp::Float64, t::Float64, σ::Float64, σp::Float64, ν::NTuple{4,Float64})
    rhs_sq = sigma_pv_rhs_sq(t, σp, σ, ν)
    rhs_sq ≤ 0 && return 0.0
    r = sqrt(rhs_sq) / max(t, eps())  # σ'' magnitude
    c1 =  sign(prev_σpp) * r
    c2 = -sign(prev_σpp) * r
    return abs(c1 - prev_σpp) ≤ abs(c2 - prev_σpp) ? c1 : c2
end

function step_sigma_pv!(σ::Float64, σp::Float64, s_now::Float64, s_next::Float64, ν::NTuple{4,Float64}, prev_σpp::Float64)
    h = s_next - s_now
    σpp = choose_sigma_pp_pv(prev_σpp, s_now, σ, σp, ν)
    σ_new  = σ  + h*σp + 0.5*h^2*σpp
    σp_mid = σp + 0.5*h*σpp
    σpp2   = choose_sigma_pp_pv(σpp, 0.5*(s_now+s_next), σ_new, σp_mid, ν)
    σp_new = σp + h*σpp2
    return σ_new, σp_new, σpp2
end

# reconstruct CDF from σ via trapezoid rule on log F
function F_from_sigma_grid(sgrid::Vector{Float64}, σvals::Vector{Float64}, s0::Float64, F0::Float64)
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
# One full run for LUE (largest eigenvalue, interval (s, ∞))
# ---------------------------
function lue_pv_locked(; N::Int=20, α::Float64=0.0, window_half::Float64=6.0*N^(1/3),
                        NquadFD::Int=260, npts::Int=1201, nanchors::Int=81)

    s_edge = (sqrt(N) + sqrt(N + α))^2     # MP top edge heuristic
    smin, smax = s_edge - window_half, s_edge + window_half
    smin = max(smin, 0.0)
    @info "LUE N=$N, α=$α  top edge≈$s_edge; domain [$smin,$smax]; Nquad=$NquadFD; anchors=$nanchors"

    # global s-grid (descending)
    sgrid = collect(range(smax, smin; length=npts))

    # anchors and local stencil
    anchors = collect(range(smax, smin; length=nanchors))
    h_anc   = (anchors[end]-anchors[1])/(length(anchors)-1)
    stencil = (-3:3) .* (0.5*h_anc)

    # Fredholm-based anchor data
    anc = Vector{NamedTuple}(undef, length(anchors))
    for (j, s0) in pairs(anchors)
        spts = Float64[s0 + δ for δ in stencil]
        Fpts = F_N_vec_LUE(N, α, spts; Nquad=NquadFD, L=tail_len_lue(N,α))
        anc[j] = sigma_from_logF(spts, Fpts)
        (j % 5 == 0) && @info "  anchor $j/$(length(anchors)) at s0=$(round(s0,digits=3))"
    end

    # integrate σ between anchors with dynamic branch; reset at anchors
    ν = (0.0, 0.0, N + α, 1.0*N)  # σ–PV parameters
    σvals  = similar(sgrid)
    σpvals = similar(sgrid)
    σvals[1]  = anc[1].σ
    σpvals[1] = anc[1].σp
    σpp_local = 0.0
    k = 1
    for j in 1:(length(anchors)-1)
        sR = anchors[j+1]
        while k < length(sgrid) && sgrid[k] > sR + 1e-12
            σvals[k+1], σpvals[k+1], σpp_local =
                step_sigma_pv!(σvals[k], σpvals[k], sgrid[k], sgrid[k+1], ν, σpp_local)
            k += 1
        end
        idx = findmin(abs.(sgrid .- sR))[2]
        σvals[idx]  = anc[j+1].σ
        σpvals[idx] = anc[j+1].σp
        k = idx
    end

    # reconstruct & compare
    F_pv = F_from_sigma_grid(sgrid, σvals, anc[1].s0, anc[1].F0)
    F_fd = F_N_vec_LUE(N, α, sgrid; Nquad=max(NquadFD,280), L=tail_len_lue(N,α))

    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-N Fredholm (Laguerre kernel)",
                xlabel="s", ylabel="CDF",
                title="LUE largest-eigenvalue CDF (N=$(N), α=$(α))")
    plot!(plt1, sgrid, F_pv, lw=2, ls=:dash, label="Painlevé V σ-form (locked)")
    savefig(plt1, "lue_pv_locked_vs_fd_N$(N)_a$(round(α,digits=3)).png"); display(plt1)

    plt2 = plot(sgrid, abs.(F_fd .- F_pv), lw=2, label="|Δ|",
                xlabel="s", ylabel="absolute error",
                title="Absolute difference: σ–PV (locked) vs Fredholm (N=$(N), α=$(α))")
    savefig(plt2, "lue_pv_locked_absdiff_N$(N)_a$(round(α,digits=3)).png"); display(plt2)

    @info "N=$N, α=$α: max |Δ| = $(maximum(abs.(F_fd .- F_pv)))"
    return (sgrid=sgrid, F_fd=F_fd, F_pv=F_pv, plt1=plt1, plt2=plt2)
end

# ---------------------------
# Run a small suite (feel free to edit)
# ---------------------------
function run_lue_suite()
    configs = [
        (N=10,  α=0.0, Nquad=220, anchors=61,  window=6.0*10^(1/3)),
        (N=20,  α=0.0, Nquad=240, anchors=81,  window=6.0*20^(1/3)),
        (N=50,  α=2.0, Nquad=260, anchors=91,  window=6.0*50^(1/3)),
        (N=100, α=5.0, Nquad=280, anchors=101, window=6.0*100^(1/3)),
    ]
    for c in configs
        lue_pv_locked(N=c.N, α=c.α, NquadFD=c.Nquad, nanchors=c.anchors, window_half=c.window)
    end
    @info "Saved all figures in $(pwd())"
end

run_lue_suite()
