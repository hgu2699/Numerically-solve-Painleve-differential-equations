###########################
# JUE hard-edge test file #
###########################

using LinearAlgebra
using SpecialFunctions
using FastGaussQuadrature
using Printf

# =============== 0. Small guard ==================

const S_GUARD = 1e-10

# =============== 1. Jacobi orthonormal basis & JUE Fredholm ===============

# log of Jacobi norm h_n for weight (1-x)^a (1+x)^b on (-1,1)
@inline function log_h_jacobi(n::Int, a::Float64, b::Float64)
    (a + b + 1.0) * log(2.0) +
    loggamma(n + a + 1.0) +
    loggamma(n + b + 1.0) -
    log(2.0*n + a + b + 1.0) -
    loggamma(n + 1.0) -
    loggamma(n + a + b + 1.0)
end

# inv sqrt norms 1/sqrt(h_n)
function inv_sqrt_h_vec(N::Int, a::Float64, b::Float64)
    v = Vector{Float64}(undef, N)
    @inbounds for n in 0:N-1
        v[n+1] = exp(-0.5 * log_h_jacobi(n, a, b))
    end
    return v
end

# One row φ_k(x) = orthonormal Jacobi polys * sqrt(weight)
function jacobi_phi_row!(
        row::AbstractVector{Float64},
        x::Float64,
        N::Int,
        a::Float64,
        b::Float64,
        inv_sqrt_h::Vector{Float64},
    )
    @assert length(row) == N
    if !(x > -1.0 && x < 1.0)
        error("Jacobi support is (-1,1). Got x=$x")
    end

    sqrtw = (1 - x)^(0.5*a) * (1 + x)^(0.5*b)

    # P_0
    Pkm1 = 1.0
    row[1] = Pkm1 * inv_sqrt_h[1] * sqrtw
    N == 1 && return

    # P_1
    Pk = 0.5 * ((a - b) + (a + b + 2.0) * x)
    row[2] = Pk * inv_sqrt_h[2] * sqrtw
    N == 2 && return

    # 3-term recurrence (DLMF 18.9.5) for P_{n+1}
    for n in 1:(N-2)
        c1 = 2.0 * (n + 1.0) * (n + a + b + 1.0) * (2.0*n + a + b)
        c2 = (2.0*n + a + b + 1.0) *
             ((2.0*n + a + b + 2.0) * (2.0*n + a + b) * x + a^2 - b^2)
        c3 = 2.0 * (n + a) * (n + b) * (2.0*n + a + b + 2.0)
        Pkp1 = (c2 * Pk - c3 * Pkm1) / c1
        Pkm1, Pk = Pk, Pkp1
        row[n+2] = Pk * inv_sqrt_h[n+2] * sqrtw
    end
end

# Φ matrix: Φ[i,k] = φ_k(t_i)
function Phi_jacobi(t::Vector{Float64}, N::Int, a::Float64, b::Float64)
    invsh = inv_sqrt_h_vec(N, a, b)
    Φ = Matrix{Float64}(undef, length(t), N)
    @inbounds for i in eachindex(t)
        jacobi_phi_row!(view(Φ, i, :), t[i], N, a, b, invsh)
    end
    return Φ
end

"""
    F_N_fredholm_JUE(N, a, b, s; Nquad=240)

Finite-N JUE largest-eigenvalue CDF

    F_N^{(J)}(s; a, b) = P(λ_max ≤ s)

for weight w(x) = (1-x)^a (1+x)^b on (-1,1), computed as
det(I - K_N) on (s,1) via Nyström/Gram.

This version:
  * uses a Gram matrix G = (WΦ)' * (WΦ) (N×N) and Sylvester's identity,
  * bumps the effective quadrature size to at least 2N for stability.
"""
function F_N_fredholm_JUE(
        N::Int,
        a::Float64,
        b::Float64,
        s::Float64;
        Nquad::Int = 240,
    )

    s_eff = min(max(s, -1.0 + S_GUARD), 1.0 - S_GUARD)
    s_eff ≤ -1.0 && return 0.0
    s_eff ≥ 1.0 && return 1.0

    # use at least 2N nodes for N up to a few hundred
    M = max(Nquad, 2N)

    # Gauss-Legendre on (s_eff,1)
    z, w = gausslegendre(M)
    t  = 0.5 .* ((1.0 - s_eff) .* z .+ (1.0 + s_eff))
    dt = 0.5 .* (1.0 - s_eff) .* w

    Φ   = Phi_jacobi(t, N, a, b)
    sqrt_dt = sqrt.(dt)
    WΦ  = Φ .* sqrt_dt               # row-scale by sqrt(dt)

    # Gram matrix G = (WΦ)' * (WΦ) (N×N)
    G   = transpose(WΦ) * WΦ

    nG  = size(G, 1)
    Mmat = Symmetric(Matrix(I, nG, nG) - G)
    λ   = eigvals(Mmat)
    λ   = clamp.(λ, eps(), 1.0)
    return exp(sum(log, λ))
end

# =============== 2. Improved Bessel hard-edge gap ===============

# Off-diagonal Bessel kernel K_Bes^{(α)}(x,y)
function bessel_kernel_offdiag(α::Float64, x::Float64, y::Float64)
    sx = sqrt(x)
    sy = sqrt(y)

    Jαx  = besselj(α, sx)
    Jαy  = besselj(α, sy)
    Jαpx = 0.5 * (besselj(α - 1.0, sx) - besselj(α + 1.0, sx))
    Jαpy = 0.5 * (besselj(α - 1.0, sy) - besselj(α + 1.0, sy))

    num = Jαx * sy * Jαpy - Jαy * sx * Jαpx
    return num / (2.0 * (x - y))
end

# Exact diagonal limit for the Bessel kernel.
# Standard identity:
#   K_α(x,x) = ( J_α(√x)^2 - J_{α-1}(√x) * J_{α+1}(√x) ) / 4
@inline function bessel_kernel_diag(α::Float64, x::Float64)
    sx   = sqrt(x)
    Jα   = besselj(α, sx)
    Jαm1 = besselj(α - 1.0, sx)
    Jαp1 = besselj(α + 1.0, sx)
    return (Jα^2 - Jαm1 * Jαp1) / 4.0
end

# Full Bessel kernel, safe on the diagonal.
function bessel_kernel(α::Float64, x::Float64, y::Float64)
    return x == y ? bessel_kernel_diag(α, x) : bessel_kernel_offdiag(α, x, y)
end

"""
    E_hard_bessel(α, s; Nquad=120)

Bessel hard-edge gap probability

    E_hard^{(α)}(s) = det(I - K_Bes^{(α)})_{L^2(0,s)}

via Nyström/Gauss-Legendre with a square-root change of variable
x = s z^2 that clusters nodes near the origin for large α.
"""
function E_hard_bessel(α::Float64, s::Float64; Nquad::Int = 120)
    s <= 0 && return 1.0

    # Gauss-Legendre on z ∈ (0,1), then x = s z^2
    ξ, ω = gausslegendre(Nquad)
    z = 0.5 .* (ξ .+ 1.0)          # (0,1)
    x = s .* (z .^ 2)              # (0,s)
    w = s .* z .* ω                # from dx = 2s z dz and dz mapping

    A = Matrix{Float64}(undef, Nquad, Nquad)
    @inbounds for i in 1:Nquad
        xi      = x[i]
        sqrtw_i = sqrt(w[i])
        for j in 1:Nquad
            A[i, j] = sqrtw_i * bessel_kernel(α, xi, x[j]) * sqrt(w[j])
        end
    end

    Mmat = Symmetric(Matrix(I, Nquad, Nquad) - A)
    λ = eigvals(Mmat)
    λ = clamp.(λ, eps(), 1.0)
    det_val = exp(sum(log, λ))
    return max(real(det_val), 0.0)
end

# =============== 3. Hard-edge scaling for JUE ===============

# Right edge t = +1, exponent a
@inline function E_hard_JUE_right(
        N::Int, a::Float64, b::Float64, s::Float64; Nquad::Int = 320
    )
    t = 1.0 - s / (2.0 * N^2)
    return F_N_fredholm_JUE(N, a, b, t; Nquad=Nquad)
end

# Left edge t = -1, exponent b; use symmetry x ↦ -x
@inline function E_hard_JUE_left(
        N::Int, a::Float64, b::Float64, s::Float64; Nquad::Int = 320
    )
    # left edge of (a,b) = right edge of (b,a)
    t_mapped = 1.0 - s / (2.0 * N^2)
    return F_N_fredholm_JUE(N, b, a, t_mapped; Nquad=Nquad)
end

# =============== 4. Driver: compare to Bessel and save plots ===============

function run_hard_edge_tests(; Ns       = [20, 40, 80, 120],
                              a::Float64 = 0.0,
                              b::Float64 = 0.0,
                              smax::Float64 = 15.0,
                              ns::Int = 200,
                              NquadFD_base::Int = 240,
                              NquadBes::Int = 120)

    sgrid = range(1e-6, smax; length = ns)

    # Right edge (t=+1): Bessel of order α = a
    F_bes_right = [E_hard_bessel(a, s; Nquad=NquadBes) for s in sgrid]

    for N in Ns
        # Heuristic: use at least max(NquadFD_base, 2N) Legendre nodes for JUE
        NquadFD = max(NquadFD_base, 2N)

        F_jue = [E_hard_JUE_right(N, a, b, s; Nquad=NquadFD) for s in sgrid]
        err   = abs.(F_jue .- F_bes_right)
        maxerr = maximum(err)
        @info "Right hard edge: N=$N, a=$a, b=$b, NquadFD=$NquadFD, max |Δ| = $maxerr"
    end

    # Left edge (t=-1): Bessel of order α = b
    F_bes_left = [E_hard_bessel(b, s; Nquad=NquadBes) for s in sgrid]

    for N in Ns
        NquadFD = max(NquadFD_base, 2N)

        F_jue = [E_hard_JUE_left(N, a, b, s; Nquad=NquadFD) for s in sgrid]
        err   = abs.(F_jue .- F_bes_left)
        maxerr = maximum(err)
        @info "Left hard edge:  N=$N, a=$a, b=$b, NquadFD=$NquadFD, max |Δ| = $maxerr"
    end

    return nothing
end

###############################
# Example: basic symmetric test
###############################

run_hard_edge_tests(Ns = [20, 40, 80, 120, 300],
                    a = 0.0, b = 0.0,
                    smax = 15.0,
                    ns = 200,
                    NquadFD_base = 240,
                    NquadBes = 120)

###############################
# Extra tests for N = 300    #
###############################

ab_list_300 = [
    (0.0, 0.0),
    (2.0, 0.0),
    (0.0, 3.0),
    (2.0, 3.0),
]

for (a,b) in ab_list_300
    @info "===== Hard-edge test (improved) for N=300, a=$a, b=$b ====="
    run_hard_edge_tests(
        Ns          = [300],     # FIX N = 300
        a           = a,
        b           = b,
        smax        = 15.0,
        ns          = 200,
        NquadFD_base = 320,      # start a bit higher for N=300
        NquadBes    = 160        # tighter Bessel quadrature
    )
end
