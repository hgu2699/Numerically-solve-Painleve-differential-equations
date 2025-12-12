# ================================================================
# lue_scaling_hard_soft_calibrated_v2.jl
#
# Hard edge:
#   LUE gap on (0, s/(4N)) → Bessel gap on (0,s).
#
# Soft edge (calibrated, left-tail-conscious):
#   1. Numerically calibrate μ_N, σ_N by matching finite-N LUE
#      to Tracy–Widom F₂ at three quantiles q = 0.1, 0.5, 0.9.
#      Uses a wide s-window and long truncation to capture left tail.
#   2. Use adaptive truncation L(s) with larger L_max, c for left tail.
#   3. Compare calibrated, scaled LUE CDF to F₂.
#
# BigFloat orthonormal Laguerre, Nyström ΦᵀΦ, Airy/Bessel kernels.
# ================================================================

using LinearAlgebra
using SpecialFunctions
using FastGaussQuadrature: gausslegendre
using Plots

# High precision for Laguerre building
setprecision(BigFloat, 256)

# ------------------------------------------------
# 1. Orthonormal Laguerre functions φ_k(x)
# ------------------------------------------------
# Weight: w(x) = x^a e^{-x}, a > -1.

"""
    phi_laguerre_all_big(N, a, x) -> Vector{Float64}

Return φ_k(x) for k = 0,…,N-1 at a given x, computed in BigFloat and
converted to Float64.
"""
function phi_laguerre_all_big(N::Int, a::Float64, x::Float64)
    N ≤ 0 && error("N must be positive")
    ab = BigFloat(a)
    xb = BigFloat(x)

    # sqrt(w(x)) = x^(a/2) e^{-x/2}
    sqrtw_b = if x == 0.0
        a == 0.0 ? BigFloat(1.0) : BigFloat(0.0)
    else
        exp((ab/2) * log(xb) - xb/2)
    end

    # p_n(x) orthonormal, p[1]=p_0,...,p[N]=p_{N-1}
    p = Vector{BigFloat}(undef, N)
    # p_0 normalized: ∫ p_0^2 w = 1 ⇒ p_0 = 1/√Γ(a+1)
    p[1] = inv(sqrt(gamma(ab + 1)))

    if N ≥ 2
        # n = 0 step: x p0 = a1 p1 + b0 p0
        a1 = sqrt((1 + ab) * 1)    # a_1
        b0 = ab + 1                # b_0
        p[2] = (xb - b0) * p[1] / a1

        # n = 1,…,N-2 recurrence
        for n in 1:(N-2)
            an1 = sqrt((n+1) * (n+1 + ab))  # a_{n+1}
            bn  = 2n + ab + 1               # b_n
            an  = sqrt(n * (n + ab))        # a_n
            p[n+2] = (xb * p[n+1] - bn * p[n+1] - an * p[n]) / an1
        end
    end

    φ = Vector{Float64}(undef, N)
    for k in 1:N
        φ[k] = Float64(p[k] * sqrtw_b)
    end
    return φ
end

# ------------------------------------------------
# 2. LUE Nyström via ΦᵀΦ
# ------------------------------------------------

"""
    fredholm_det_LUE_interval(N, a, a0, b0; n=80)

Nyström determinant det(I - K_Laguerre) on [a0,b0] using n-point Gauss–Legendre.
K is built as K = ΦᵀΦ with Φ_{k,i} = φ_k(x_i).
"""
function fredholm_det_LUE_interval(N::Int, a::Float64,
                                   a0::Float64, b0::Float64;
                                   n::Int=80)
    z, w = gausslegendre(n)
    xs = 0.5*(b0-a0) .* (z .+ 1.0) .+ 0.5*(a0+b0)
    ws = 0.5*(b0-a0) .* w
    n_nodes = length(xs)

    Φ = Matrix{Float64}(undef, N, n_nodes)
    for j in 1:n_nodes
        φ = phi_laguerre_all_big(N, a, xs[j])
        @inbounds for k in 1:N
            Φ[k,j] = φ[k]
        end
    end

    K_raw = Φ' * Φ
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(ws[i]) * K_raw[i,j] * sqrt(ws[j])
    end

    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

"""
    fredholm_det_LUE_semiinfinite(N, a, s; L=12.0, n=240)

Nyström determinant det(I - K_Laguerre) on (s,∞)
truncated to (s, s+L) and mapped to [0,L].
"""
function fredholm_det_LUE_semiinfinite(N::Int, a::Float64, s::Float64;
                                       L::Float64=12.0, n::Int=240)
    z, w = gausslegendre(n)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u
    n_nodes = length(t)

    Φ = Matrix{Float64}(undef, N, n_nodes)
    for j in 1:n_nodes
        φ = phi_laguerre_all_big(N, a, t[j])
        @inbounds for k in 1:N
            Φ[k,j] = φ[k]
        end
    end

    K_raw = Φ' * Φ
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(dt[i]) * K_raw[i,j] * sqrt(dt[j])
    end

    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

# ------------------------------------------------
# 3. Bessel / Airy kernels + generic Fredholm
# ------------------------------------------------

function KBessel(a::Float64, x::Float64, y::Float64; eps::Float64=1e-8)
    sx, sy = sqrt(x), sqrt(y)
    if abs(x-y) > eps
        num = sx * besselj(a+1.0, sx) * besselj(a, sy) -
              sy * besselj(a+1.0, sy) * besselj(a, sx)
        return num / (2.0*(x-y))
    else
        J  = besselj(a, sx)
        Jm = besselj(a-1.0, sx)
        Jp = besselj(a+1.0, sx)
        return (J^2 - Jm*Jp) / 4.0
    end
end

function KAiry(s::Float64, t::Float64; eps::Float64=1e-8)
    if abs(s - t) > eps
        Ai_s  = airyai(s);   Ai_t  = airyai(t)
        Aip_s = airyaiprime(s); Aip_t = airyaiprime(t)
        num = Ai_s * Aip_t - Ai_t * Aip_s
        return num / (s - t)
    else
        Ai_s  = airyai(s)
        Aip_s = airyaiprime(s)
        return Aip_s^2 - s * Ai_s^2
    end
end

function fredholm_det_interval(kernel::Function, a::Float64, b::Float64; n::Int=80)
    x, w = gausslegendre(n)
    xs = 0.5*(b-a) .* (x .+ 1.0) .+ 0.5*(a+b)
    ws = 0.5*(b-a) .* w

    n_nodes = length(xs)
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(ws[i]) * kernel(xs[i], xs[j]) * sqrt(ws[j])
    end
    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

function fredholm_det_semiinfinite(kernel::Function, s::Float64;
                                   L::Float64=12.0, n::Int=140)
    z, w = gausslegendre(n)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u

    n_nodes = length(t)
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(dt[i]) * kernel(t[i], t[j]) * sqrt(dt[j])
    end
    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

# ------------------------------------------------
# 4. HARD EDGE: LUE → Bessel
# ------------------------------------------------

function gap_LUE_hard(N::Int, a::Float64, s::Float64; n_quad::Int=80)
    t = s / (4.0 * N)
    return fredholm_det_LUE_interval(N, a, 0.0, t; n=n_quad)
end

function gap_Bessel(a::Float64, s::Float64; n_quad::Int=80)
    kern(x,y) = KBessel(a, x, y)
    return fredholm_det_interval(kern, 0.0, s; n=n_quad)
end

function test_hard_edge(; a::Float64=0.0,
                         Ns = [20, 40, 80],
                         s_grid = range(0.5, 10.0; length=8),
                         n_quad::Int=80)
    println("====================================================")
    println("Hard-edge scaling test: LUE → Bessel (a = $a)")
    println("Scaling t = s/(4N); compare E_N^{(L)}((0,t);1) to Bessel gap on (0,s).")
    println("====================================================")
    for N in Ns
        println("\n--- N = $N ---")
        maxdiff = 0.0
        for s in s_grid
            F_LUE = gap_LUE_hard(N, a, s; n_quad=n_quad)
            F_Bes = gap_Bessel(a, s;    n_quad=n_quad)
            diff  = F_LUE - F_Bes
            maxdiff = max(maxdiff, abs(diff))
            println("s = $(round(s,digits=3))  F_LUE = $(F_LUE),  F_Bes = $(F_Bes),  diff = $(diff)")
        end
        println("Max |diff| for N=$N: $maxdiff")
    end
end

# ------------------------------------------------
# 5. Tracy–Widom F₂ via Airy Fredholm + quantile helper
# ------------------------------------------------

function F_TW2(x::Float64; L_Airy::Float64=12.0, n::Int=160)
    kern(s,t) = KAiry(s, t)
    return fredholm_det_semiinfinite(kern, x; L=L_Airy, n=n)
end

"""
    TW_quantile(q; xL=-15, xR=15, tol=1e-6)

Find x such that F₂(x) ≈ q by bisection.
"""
function TW_quantile(q::Float64; xL::Float64=-15.0, xR::Float64=15.0, tol::Float64=1e-6)
    left, right = xL, xR
    for _ in 1:80
        mid = 0.5*(left+right)
        Fmid = F_TW2(mid)
        if Fmid < q
            left = mid
        else
            right = mid
        end
        if abs(right-left) < tol
            break
        end
    end
    return 0.5*(left+right)
end

# ------------------------------------------------
# 6. LUE largest eigenvalue: fixed-L & calibrated version
# ------------------------------------------------

"""
    F_LUE_fixed(N, a, s; L=80, n=260)

Largest-eigenvalue CDF using a fixed truncation (s, s+L).
Used only for μ,σ calibration.  Here L is big to capture left tail.
"""
function F_LUE_fixed(N::Int, a::Float64, s::Float64;
                     L::Float64=80.0, n::Int=260)
    return fredholm_det_LUE_semiinfinite(N, a, s; L=L, n=n)
end

"""
    calibrate_mu_sigma_LUE(N, a; qs=[0.1,0.5,0.9])

Numerically find μ_N, σ_N so that the finite-N LUE CDF matches
Tracy–Widom F₂ at quantiles qs (three points, including left tail).
Uses least-squares fit of s ≈ μ_N + σ_N x over the matched pairs.
Includes clamping of qs into the LUE CDF range to avoid errors.
"""
function calibrate_mu_sigma_LUE(N::Int, a::Float64;
                                qs::Vector{Float64} = [0.1, 0.5, 0.9])
    # --- Step 0: wide s-grid around MP edge to see left tail ---
    s_edge = (sqrt(N + a) + sqrt(N))^2
    smin = s_edge - 60.0    # far enough into the left tail
    smax = s_edge + 20.0
    n_s  = 201
    s_grid = collect(range(smin, smax; length=n_s))

    # LUE CDF on this window, with a long truncation
    F_vals = [F_LUE_fixed(N, a, s; L=80.0, n=260) for s in s_grid]

    Fmin, Fmax = minimum(F_vals), maximum(F_vals)
    δ = 1e-4

    # --- Step 1: clamp target quantiles into [Fmin,Fmax] ---
    q_eff = [clamp(q, Fmin + δ, Fmax - δ) for q in qs]
    if any(q_eff .!= qs)
        println("[Calibration] requested qs = $(qs), " *
                "but LUE CDF on [smin,smax] has range [$Fmin, $Fmax].")
        println("[Calibration] adjusted to q_eff = $(q_eff).")
    end

    # --- Step 2: TW quantiles for the adjusted q's ---
    xq = [TW_quantile(q) for q in q_eff]

    # --- helper: find s where F_LUE ≈ q via interpolation ---
    function find_s_for_q(q::Float64)
        for k in 1:(n_s-1)
            Fk, Fk1 = F_vals[k], F_vals[k+1]
            if (Fk ≤ q ≤ Fk1) || (Fk1 ≤ q ≤ Fk)
                t = (q - Fk) / (Fk1 - Fk)
                return s_grid[k] + t*(s_grid[k+1] - s_grid[k])
            end
        end
        # Should not happen after clamp, but be safe:
        return q < Fmin ? s_grid[1] : s_grid[end]
    end

    s_q = [find_s_for_q(q) for q in q_eff]

    # --- Step 3: least-squares fit s ≈ μ + σ x ---
    M = hcat(ones(length(xq)), xq)
    θ = M \ s_q
    μN, σN = θ[1], θ[2]

    println("\n[Calibration] using q_eff = $(q_eff)")
    println("[Calibration] xq (TW) = $(xq)")
    println("[Calibration] s_q (LUE) = $(s_q)")
    println("[Calibration] μ_N ≈ $μN, σ_N ≈ $σN")

    return μN, σN, xq, s_q
end

"""
    F_LUE_largest_scaled(N, a, s; μN, σN, c=10.0, Lmax=80, n=240)

Largest-eigenvalue CDF with adaptive truncation:
L(s) = min(Lmax, μN + cσN - s), clipped to >= 4.
Lmax and c are chosen fairly large to better resolve the left tail.
"""
function F_LUE_largest_scaled(N::Int, a::Float64, s::Float64;
                              μN::Float64, σN::Float64,
                              c::Float64=10.0, Lmax::Float64=80.0,
                              n::Int=240)
    L = (μN + c*σN) - s
    if L < 4.0
        L = 4.0
    end
    L = min(L, Lmax)
    return fredholm_det_LUE_semiinfinite(N, a, s; L=L, n=n)
end

# ------------------------------------------------
# 7. Soft-edge calibrated test
# ------------------------------------------------

function test_soft_edge_calibrated(N::Int, a::Float64;
                                   qs::Vector{Float64} = [0.1, 0.5, 0.9],
                                   x_min::Float64 = -6.0,
                                   x_max::Float64 = 6.0,
                                   nx::Int = 61)
    println("\n====================================================")
    println("Soft-edge scaling test (calibrated): LUE → TW F₂")
    println("N = $N, a = $a")
    println("====================================================")

    μN, σN, xq, s_q = calibrate_mu_sigma_LUE(N, a; qs=qs)

    xgrid = collect(range(x_min, x_max; length=nx))
    sgrid = μN .+ σN .* xgrid

    F_LUE_vals = [F_LUE_largest_scaled(N, a, s;
                                       μN=μN, σN=σN,
                                       c=10.0, Lmax=80.0, n=240)
                  for s in sgrid]
    F_TW_vals  = [F_TW2(x; L_Airy=12.0, n=160) for x in xgrid]

    # CDF comparison
    pltCDF = plot(xgrid, F_LUE_vals, lw=2, label="LUE N=$N, a=$a (scaled)",
                  xlabel="x", ylabel="CDF",
                  title="Soft edge (calibrated): LUE vs TW F₂ (N=$N)")
    plot!(pltCDF, xgrid, F_TW_vals, lw=2, ls=:dash, label="Tracy–Widom F₂")
    savefig(pltCDF, "soft_edge_calibrated_LUE_vs_TW2_N$(N)_a$(round(a,digits=2)).png")
    display(pltCDF)

    # Error
    err = abs.(F_LUE_vals .- F_TW_vals)
    pltErr = plot(xgrid, err, lw=2, label="|F_LUE - F₂|",
                  xlabel="x", ylabel="|Δ|",
                  title="Soft-edge error (calibrated, N=$N, a=$a)")
    savefig(pltErr, "soft_edge_calibrated_error_N$(N)_a$(round(a,digits=2)).png")
    display(pltErr)

    println("Max |F_LUE - F₂| for N=$N, a=$a: $(maximum(err))")
    return (xgrid=xgrid, F_LUE=F_LUE_vals, F_TW=F_TW_vals, err=err,
            μN=μN, σN=σN, xq=xq, s_q=s_q)
end

# ------------------------------------------------
# 8. Main
# ------------------------------------------------

function main()
    # Hard edge: LUE → Bessel
    test_hard_edge(a=0.0, Ns=[20,40,80],
                   s_grid=range(0.5,10.0;length=8), n_quad=80)

    # Soft edge, calibrated with special care for left tail.
    # You can change N_soft if runtime is too heavy.
    N_soft = 500
    test_soft_edge_calibrated(N_soft, 0.0; qs=[0.1, 0.5, 0.9],
                              x_min=-6.0, x_max=6.0, nx=61)
end

main()
# ================================================================
# lue_scaling_with_hard_edge_plots.jl
#
# Hard edge:
#   LUE gap on (0, s/(4N)) → Bessel gap on (0,s).
#   Now with CDF + error plots for each N.
#
# Soft edge (calibrated):
#   LUE largest eigenvalue → Tracy–Widom F₂, with μ_N, σ_N fitted
#   from quantiles q = 0.1, 0.5, 0.9.
#
# This file is self-contained.
# ================================================================

using LinearAlgebra
using SpecialFunctions
using FastGaussQuadrature: gausslegendre
using Plots

# High precision for Laguerre building
setprecision(BigFloat, 256)

# ------------------------------------------------
# 1. Orthonormal Laguerre functions φ_k(x)
# ------------------------------------------------

"""
    phi_laguerre_all_big(N, a, x) -> Vector{Float64}

Return φ_k(x) for k = 0,…,N-1 at a given x, computed in BigFloat and
converted to Float64.
"""
function phi_laguerre_all_big(N::Int, a::Float64, x::Float64)
    N ≤ 0 && error("N must be positive")
    ab = BigFloat(a)
    xb = BigFloat(x)

    # sqrt(w(x)) = x^(a/2) e^{-x/2}, w(x)=x^a e^{-x}
    sqrtw_b = if x == 0.0
        a == 0.0 ? BigFloat(1.0) : BigFloat(0.0)
    else
        exp((ab/2) * log(xb) - xb/2)
    end

    # p_n(x) orthonormal, p[1]=p_0,...,p[N]=p_{N-1}
    p = Vector{BigFloat}(undef, N)
    # p_0 normalized: ∫ p_0^2 w = 1 ⇒ p_0 = 1/√Γ(a+1)
    p[1] = inv(sqrt(gamma(ab + 1)))

    if N ≥ 2
        # n = 0 step: x p0 = a1 p1 + b0 p0
        a1 = sqrt((1 + ab) * 1)    # a_1
        b0 = ab + 1                # b_0
        p[2] = (xb - b0) * p[1] / a1

        # n = 1,…,N-2 recurrence
        for n in 1:(N-2)
            an1 = sqrt((n+1) * (n+1 + ab))  # a_{n+1}
            bn  = 2n + ab + 1               # b_n
            an  = sqrt(n * (n + ab))        # a_n
            p[n+2] = (xb * p[n+1] - bn * p[n+1] - an * p[n]) / an1
        end
    end

    φ = Vector{Float64}(undef, N)
    for k in 1:N
        φ[k] = Float64(p[k] * sqrtw_b)
    end
    return φ
end

# ------------------------------------------------
# 2. LUE Nyström via ΦᵀΦ
# ------------------------------------------------

"""
    fredholm_det_LUE_interval(N, a, a0, b0; n=80)

Nyström determinant det(I - K_Laguerre) on [a0,b0] using n-point
Gauss–Legendre. K is built as K = ΦᵀΦ with Φ_{k,i} = φ_k(x_i).
"""
function fredholm_det_LUE_interval(N::Int, a::Float64,
                                   a0::Float64, b0::Float64;
                                   n::Int=80)
    z, w = gausslegendre(n)
    xs = 0.5*(b0-a0) .* (z .+ 1.0) .+ 0.5*(a0+b0)
    ws = 0.5*(b0-a0) .* w
    n_nodes = length(xs)

    Φ = Matrix{Float64}(undef, N, n_nodes)
    for j in 1:n_nodes
        φ = phi_laguerre_all_big(N, a, xs[j])
        @inbounds for k in 1:N
            Φ[k,j] = φ[k]
        end
    end

    K_raw = Φ' * Φ
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(ws[i]) * K_raw[i,j] * sqrt(ws[j])
    end

    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

"""
    fredholm_det_LUE_semiinfinite(N, a, s; L=12.0, n=240)

Nyström determinant det(I - K_Laguerre) on (s,∞), truncated to (s,s+L).
"""
function fredholm_det_LUE_semiinfinite(N::Int, a::Float64, s::Float64;
                                       L::Float64=12.0, n::Int=240)
    z, w = gausslegendre(n)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u
    n_nodes = length(t)

    Φ = Matrix{Float64}(undef, N, n_nodes)
    for j in 1:n_nodes
        φ = phi_laguerre_all_big(N, a, t[j])
        @inbounds for k in 1:N
            Φ[k,j] = φ[k]
        end
    end

    K_raw = Φ' * Φ
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(dt[i]) * K_raw[i,j] * sqrt(dt[j])
    end

    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

# ------------------------------------------------
# 3. Bessel / Airy kernels + generic Fredholm
# ------------------------------------------------

function KBessel(a::Float64, x::Float64, y::Float64; eps::Float64=1e-8)
    sx, sy = sqrt(x), sqrt(y)
    if abs(x-y) > eps
        num = sx * besselj(a+1.0, sx) * besselj(a, sy) -
              sy * besselj(a+1.0, sy) * besselj(a, sx)
        return num / (2.0*(x-y))
    else
        J  = besselj(a, sx)
        Jm = besselj(a-1.0, sx)
        Jp = besselj(a+1.0, sx)
        return (J^2 - Jm*Jp) / 4.0
    end
end

function KAiry(s::Float64, t::Float64; eps::Float64=1e-8)
    if abs(s - t) > eps
        Ai_s  = airyai(s);   Ai_t  = airyai(t)
        Aip_s = airyaiprime(s); Aip_t = airyaiprime(t)
        num = Ai_s * Aip_t - Ai_t * Aip_s
        return num / (s - t)
    else
        Ai_s  = airyai(s)
        Aip_s = airyaiprime(s)
        return Aip_s^2 - s * Ai_s^2
    end
end

function fredholm_det_interval(kernel::Function, a::Float64, b::Float64; n::Int=80)
    x, w = gausslegendre(n)
    xs = 0.5*(b-a) .* (x .+ 1.0) .+ 0.5*(a+b)
    ws = 0.5*(b-a) .* w

    n_nodes = length(xs)
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(ws[i]) * kernel(xs[i], xs[j]) * sqrt(ws[j])
    end
    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

function fredholm_det_semiinfinite(kernel::Function, s::Float64;
                                   L::Float64=12.0, n::Int=140)
    z, w = gausslegendre(n)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u

    n_nodes = length(t)
    Kmat = Matrix{Float64}(undef, n_nodes, n_nodes)
    for i in 1:n_nodes, j in 1:n_nodes
        Kmat[i,j] = sqrt(dt[i]) * kernel(t[i], t[j]) * sqrt(dt[j])
    end
    M = Matrix{Float64}(I, n_nodes, n_nodes) - Kmat
    λ = eigvals(M)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

# ------------------------------------------------
# 4. HARD EDGE: LUE → Bessel, with plots
# ------------------------------------------------

function gap_LUE_hard(N::Int, a::Float64, s::Float64; n_quad::Int=80)
    t = s / (4.0 * N)
    return fredholm_det_LUE_interval(N, a, 0.0, t; n=n_quad)
end

function gap_Bessel(a::Float64, s::Float64; n_quad::Int=80)
    kern(x,y) = KBessel(a, x, y)
    return fredholm_det_interval(kern, 0.0, s; n=n_quad)
end

"""
    test_hard_edge_with_plots(...)

Prints numeric comparison and produces, for each N:
  * CDF plot: LUE vs Bessel gap,
  * error plot: |F_LUE - F_Bes|.
"""
function test_hard_edge_with_plots(; a::Float64=0.0,
                                   Ns = [20, 40, 80],
                                   s_grid_print = range(0.5, 10.0; length=8),
                                   s_min::Float64 = 0.5,
                                   s_max::Float64 = 10.0,
                                   ns_plot::Int = 200,
                                   n_quad::Int=80)

    println("====================================================")
    println("Hard-edge scaling test: LUE → Bessel (a = $a)")
    println("Scaling t = s/(4N); compare E_N^{(L)}((0,t);1) to Bessel gap on (0,s).")
    println("====================================================")

    # Plot grid and Bessel limit once (independent of N)
    sgrid_plot = collect(range(s_min, s_max; length=ns_plot))
    F_Bes_plot = [gap_Bessel(a, s; n_quad=n_quad) for s in sgrid_plot]

    for N in Ns
        println("\n--- N = $N ---")
        maxdiff_print = 0.0
        for s in s_grid_print
            F_LUE = gap_LUE_hard(N, a, s; n_quad=n_quad)
            F_Bes = gap_Bessel(a, s;    n_quad=n_quad)
            diff  = F_LUE - F_Bes
            maxdiff_print = max(maxdiff_print, abs(diff))
            println("s = $(round(s,digits=3))  F_LUE = $(F_LUE),  F_Bes = $(F_Bes),  diff = $(diff)")
        end
        println("Max |diff| for N=$N (print grid): $maxdiff_print")

        # Plot data on finer grid
        F_LUE_plot = [gap_LUE_hard(N, a, s; n_quad=n_quad) for s in sgrid_plot]
        diff_plot  = F_LUE_plot .- F_Bes_plot

        # CDF comparison
        pltCDF = plot(sgrid_plot, F_Bes_plot, lw=2, label="Bessel gap (limit)",
                      xlabel="s", ylabel="CDF",
                      title="Hard edge: LUE (N=$N,a=$a) vs Bessel")
        plot!(pltCDF, sgrid_plot, F_LUE_plot, lw=2, ls=:dash,
              label="LUE gap, N=$N")
        savefig(pltCDF, "hard_edge_LUE_vs_Bessel_N$(N)_a$(round(a,digits=2)).png")
        display(pltCDF)

        # Error plot
        pltErr = plot(sgrid_plot, abs.(diff_plot), lw=2,
                      label="|F_LUE - F_Bes|",
                      xlabel="s", ylabel="|Δ|",
                      title="Hard-edge error (N=$N, a=$a)")
        savefig(pltErr, "hard_edge_error_N$(N)_a$(round(a,digits=2)).png")
        display(pltErr)

        println("Max |diff| for N=$N (plot grid): $(maximum(abs.(diff_plot)))")
    end
end

# ------------------------------------------------
# 5. Tracy–Widom F₂ via Airy Fredholm + quantile helper
# ------------------------------------------------

function F_TW2(x::Float64; L_Airy::Float64=12.0, n::Int=160)
    kern(s,t) = KAiry(s, t)
    return fredholm_det_semiinfinite(kern, x; L=L_Airy, n=n)
end

"""
    TW_quantile(q; xL=-15, xR=15, tol=1e-6)

Find x such that F₂(x) ≈ q by bisection.
"""
function TW_quantile(q::Float64; xL::Float64=-15.0, xR::Float64=15.0, tol::Float64=1e-6)
    left, right = xL, xR
    for _ in 1:80
        mid = 0.5*(left+right)
        Fmid = F_TW2(mid)
        if Fmid < q
            left = mid
        else
            right = mid
        end
        if abs(right-left) < tol
            break
        end
    end
    return 0.5*(left+right)
end

# ------------------------------------------------
# 6. LUE largest eigenvalue: fixed-L & calibrated version
# ------------------------------------------------

"""
    F_LUE_fixed(N, a, s; L=80, n=260)

Largest-eigenvalue CDF using a fixed truncation (s, s+L).
Used only for μ,σ calibration. L is large to capture left tail.
"""
function F_LUE_fixed(N::Int, a::Float64, s::Float64;
                     L::Float64=80.0, n::Int=260)
    return fredholm_det_LUE_semiinfinite(N, a, s; L=L, n=n)
end

"""
    calibrate_mu_sigma_LUE(N, a; qs=[0.1,0.5,0.9])

Numerically find μ_N, σ_N so that the finite-N LUE CDF matches
Tracy–Widom F₂ at quantiles qs.
"""
function calibrate_mu_sigma_LUE(N::Int, a::Float64;
                                qs::Vector{Float64} = [0.1, 0.5, 0.9])

    # Wide s-grid around MP edge to see left tail
    s_edge = (sqrt(N + a) + sqrt(N))^2
    smin = s_edge - 60.0
    smax = s_edge + 20.0
    n_s  = 201
    s_grid = collect(range(smin, smax; length=n_s))

    # LUE CDF on this window, with a long truncation
    F_vals = [F_LUE_fixed(N, a, s; L=80.0, n=260) for s in s_grid]

    Fmin, Fmax = minimum(F_vals), maximum(F_vals)
    δ = 1e-4

    # Clamp target quantiles into [Fmin,Fmax]
    q_eff = [clamp(q, Fmin + δ, Fmax - δ) for q in qs]
    if any(q_eff .!= qs)
        println("[Calibration] requested qs = $(qs), " *
                "but LUE CDF on [smin,smax] has range [$Fmin, $Fmax].")
        println("[Calibration] adjusted to q_eff = $(q_eff).")
    end

    # TW quantiles for the adjusted q's
    xq = [TW_quantile(q) for q in q_eff]

    # helper: find s where F_LUE ≈ q via interpolation
    function find_s_for_q(q::Float64)
        for k in 1:(n_s-1)
            Fk, Fk1 = F_vals[k], F_vals[k+1]
            if (Fk ≤ q ≤ Fk1) || (Fk1 ≤ q ≤ Fk)
                t = (q - Fk) / (Fk1 - Fk)
                return s_grid[k] + t*(s_grid[k+1] - s_grid[k])
            end
        end
        return q < Fmin ? s_grid[1] : s_grid[end]
    end

    s_q = [find_s_for_q(q) for q in q_eff]

    # least-squares fit s ≈ μ + σ x
    M = hcat(ones(length(xq)), xq)
    θ = M \ s_q
    μN, σN = θ[1], θ[2]

    println("\n[Calibration] using q_eff = $(q_eff)")
    println("[Calibration] xq (TW) = $(xq)")
    println("[Calibration] s_q (LUE) = $(s_q)")
    println("[Calibration] μ_N ≈ $μN, σ_N ≈ $σN")

    return μN, σN, xq, s_q
end

"""
    F_LUE_largest_scaled(N, a, s; μN, σN, c=10.0, Lmax=80, n=240)

Largest-eigenvalue CDF with adaptive truncation:
L(s) = min(Lmax, μN + cσN - s), clipped to >= 4.
"""
function F_LUE_largest_scaled(N::Int, a::Float64, s::Float64;
                              μN::Float64, σN::Float64,
                              c::Float64=10.0, Lmax::Float64=80.0,
                              n::Int=240)
    L = (μN + c*σN) - s
    if L < 4.0
        L = 4.0
    end
    L = min(L, Lmax)
    return fredholm_det_LUE_semiinfinite(N, a, s; L=L, n=n)
end

# ------------------------------------------------
# 7. Soft-edge calibrated test
# ------------------------------------------------

function test_soft_edge_calibrated(N::Int, a::Float64;
                                   qs::Vector{Float64} = [0.1, 0.5, 0.9],
                                   x_min::Float64 = -6.0,
                                   x_max::Float64 = 6.0,
                                   nx::Int = 61)
    println("\n====================================================")
    println("Soft-edge scaling test (calibrated): LUE → TW F₂")
    println("N = $N, a = $a")
    println("====================================================")

    μN, σN, xq, s_q = calibrate_mu_sigma_LUE(N, a; qs=qs)

    xgrid = collect(range(x_min, x_max; length=nx))
    sgrid = μN .+ σN .* xgrid

    F_LUE_vals = [F_LUE_largest_scaled(N, a, s;
                                       μN=μN, σN=σN,
                                       c=10.0, Lmax=80.0, n=240)
                  for s in sgrid]
    F_TW_vals  = [F_TW2(x; L_Airy=12.0, n=160) for x in xgrid]

    # CDF comparison
    pltCDF = plot(xgrid, F_LUE_vals, lw=2, label="LUE N=$N, a=$a (scaled)",
                  xlabel="x", ylabel="CDF",
                  title="Soft edge (calibrated): LUE vs TW F₂ (N=$N)")
    plot!(pltCDF, xgrid, F_TW_vals, lw=2, ls=:dash, label="Tracy–Widom F₂")
    savefig(pltCDF, "soft_edge_calibrated_LUE_vs_TW2_N$(N)_a$(round(a,digits=2)).png")
    display(pltCDF)

    # Error
    err = abs.(F_LUE_vals .- F_TW_vals)
    pltErr = plot(xgrid, err, lw=2, label="|F_LUE - F₂|",
                  xlabel="x", ylabel="|Δ|",
                  title="Soft-edge error (calibrated, N=$N, a=$a)")
    savefig(pltErr, "soft_edge_calibrated_error_N$(N)_a$(round(a,digits=2)).png")
    display(pltErr)

    println("Max |F_LUE - F₂| for N=$N, a=$a: $(maximum(err))")
    return (xgrid=xgrid, F_LUE=F_LUE_vals, F_TW=F_TW_vals, err=err,
            μN=μN, σN=σN, xq=xq, s_q=s_q)
end

# ------------------------------------------------
# 8. Main
# ------------------------------------------------

function main()
    # Hard edge: print + plots
    test_hard_edge_with_plots(a=0.0,
                              Ns=[20,40,80],
                              s_grid_print=range(0.5,10.0;length=8),
                              s_min=0.5, s_max=10.0,
                              ns_plot=200, n_quad=80)

    # Soft edge: calibrated LUE → TW F₂ (same as before)
    N_soft = 500
    test_soft_edge_calibrated(N_soft, 0.0;
                              qs=[0.1, 0.5, 0.9],
                              x_min=-6.0, x_max=6.0, nx=61)
end

main()
