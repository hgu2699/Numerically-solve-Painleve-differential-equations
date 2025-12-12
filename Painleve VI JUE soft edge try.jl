###############################################################
# jue_soft_edge_deterministic.jl
#
# Deterministic check that the normalised largest eigenvalue
# of a Jacobi unitary ensemble (double Wishart / MANOVA
# regime, via orthogonal polynomial kernel) converges in law
# to Tracy–Widom F₂ under Johnstone's logit scaling.
#
# No Monte Carlo. No jacobiP dependency. No Unicode vars.
###############################################################

#########################
# 0. Package bootstrap  #
#########################

import Pkg
for p in ["FastGaussQuadrature", "SpecialFunctions"]
    if Base.find_package(p) === nothing
        try
            Pkg.add(p)
        catch
            Pkg.Registry.update()
            Pkg.add(p)
        end
    end
end

using FastGaussQuadrature          # gausslegendre
using SpecialFunctions             # airyai, airyaiprime, loggamma
using LinearAlgebra

const Ai  = airyai
const Aip = airyaiprime

###############################################################
# 1. Tracy–Widom F₂ via Airy kernel (Fredholm determinant)
###############################################################

# Airy kernel:
#   K_Airy(s,t) = (Ai(s)Ai'(t) - Ai'(s)Ai(t))/(s - t)
# with diagonal limit K_Airy(s,s) = Ai'(s)^2 - s Ai(s)^2
function airy_kernel(s::Float64, t::Float64)
    if s == t
        return Aip(s)^2 - s*Ai(s)^2
    else
        return (Ai(s)*Aip(t) - Aip(s)*Ai(t)) / (s - t)
    end
end

"""
    F2_fredholm(x; N=80)

Tracy–Widom F₂ CDF at x via Nyström method for the Airy kernel
on (x,∞):

- Map z∈(0,1) → t = x + z/(1-z)  (compresses [0,∞) to (0,1))
- Use Gauss–Legendre on z∈(0,1)
"""
function F2_fredholm(x::Float64; N::Int=80)
    # Gauss–Legendre nodes/weights on (-1,1)
    z, w_leg = gausslegendre(N)
    # Map to (0,1)
    z  = (z .+ 1.0) ./ 2.0
    w_leg  = w_leg ./ 2.0

    # Map to (x,∞)
    t  = x .+ z ./ (1 .- z)
    dt = w_leg ./ (1 .- z).^2

    sqrt_dt = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)

    @inbounds for i in 1:N
        ti = t[i]
        for j in 1:N
            tj = t[j]
            Kij = (i == j) ? (Aip(ti)^2 - ti*Ai(ti)^2) :
                             ((Ai(ti)*Aip(tj) - Aip(ti)*Ai(tj)) / (ti - tj))
            A[i,j] = sqrt_dt[i] * Kij * sqrt_dt[j]
        end
    end

    # B = I - A, determinant via eigenvalues
    B = Symmetric(Matrix(I, N, N) - A)
    λ = eigvals(B)
    λ = clamp.(real.(λ), eps(), 1.0)  # numerical safety
    return exp(sum(log, λ))
end

F2_fredholm_vec(xs::AbstractVector{<:Real}; N::Int=80) =
    [F2_fredholm(Float64(x); N=N) for x in xs]

# simple linear interpolation on a grid
function interp1(x::Vector{Float64}, y::Vector{Float64}, xi::Vector{Float64})
    yi = similar(xi)
    j = 1
    for k in eachindex(xi)
        xk = xi[k]
        while j < length(x) && x[j+1] < xk
            j += 1
        end
        if xk <= x[1]
            yi[k] = y[1]
        elseif xk >= x[end]
            yi[k] = y[end]
        else
            t = (xk - x[j]) / (x[j+1] - x[j])
            yi[k] = (1-t)*y[j] + t*y[j+1]
        end
    end
    return yi
end

# approximate mean and std of F₂ from tabulated CDF
function tw_moments_from_cdf(x::Vector{Float64}, F::Vector{Float64})
    @assert length(x) == length(F)
    p    = diff(F)
    xmid = @. (x[1:end-1] + x[2:end]) / 2
    μ    = sum(p .* xmid)
    σ2   = sum(p .* (xmid .- μ).^2)
    return μ, sqrt(σ2)
end

###############################################################
# 2. Johnstone's logit-centering for double Wishart / JUE
###############################################################

"""
    johnstone_complex_mu_sigma(p, m, n)

Complex double-Wishart centering μ_C and scaling σ_C for

  A ∼ CW_p(I, m), B ∼ CW_p(I, n),
  θ_max = largest eigenvalue of (A+B)⁻¹ B.

This is the MANOVA/JUE soft-edge scaling from
Johnstone, "Multivariate analysis and Jacobi ensembles:
largest eigenvalue and Tracy–Widom limits", 2008.
"""
function johnstone_complex_mu_sigma(p::Int, m::Int, n::Int)
    N = min(n, p)
    α = m - p
    β = abs(n - p)

    function local_params(N::Int)
        denom = 2N + α + β + 1
        s2γ = (N + 0.5) / denom
        s2ϕ = (N + β + 0.5) / denom
        # clamp to avoid slight negatives
        s2γ = min(max(s2γ, 0.0), 1.0)
        s2ϕ = min(max(s2ϕ, 0.0), 1.0)

        γN = 2 * asin(sqrt(s2γ))
        ϕN = 2 * asin(sqrt(s2ϕ))

        wN = 2 * log(tan((ϕN + γN) / 2))
        ω3 = 16.0 / (denom^2 * (sin(ϕN + γN)^2 * sin(ϕN) * sin(γN)))
        τN = ω3^(1/3)
        return wN, τN
    end

    wN,  τN  = local_params(N)
    wNm1,τNm1 = local_params(N - 1)

    μC = (wN/τN + wNm1/τNm1) / (1/τN + 1/τNm1)
    σC_inv = 0.5 * (1/τN + 1/τNm1)
    σC = 1 / σC_inv

    return μC, σC
end

logit(x)    = log(x / (1 - x))
logistic(w) = exp(w) / (1 + exp(w))

###############################################################
# 3. Jacobi polynomials and JUE kernel (deterministic)
###############################################################

# ---------------------------------------------------------
# Local implementation of Jacobi polynomials J_n^{(α,β)}(x)
# via 3-term recurrence (standard NIST formula):
#
# J_0(x) = 1
# J_1(x) = 0.5*( (α+β+2)x + (α-β) )
#
# J_{n+1}(x) = (a_n x - b_n) J_n(x) - c_n J_{n-1}(x), n≥1
#
# with
#   a_n = (2n+α+β+1)(2n+α+β+2) / [2(n+1)(n+α+β+1)]
#   b_n = (β^2-α^2)(2n+α+β+1) / [2(n+1)(n+α+β+1)(2n+α+β)]
#   c_n = (n+α)(n+β)(2n+α+β+2) / [(n+1)(n+α+β+1)(2n+α+β)]
#
# Returns J_0,...,J_{N-1} at a fixed x as a length-N vector.
# ---------------------------------------------------------
function jacobiP_eval_all(N::Int, α::Float64, β::Float64, x::Float64)
    P = zeros(Float64, N)
    if N == 0
        return P
    end

    # n = 0
    P[1] = 1.0
    if N == 1
        return P
    end

    # n = 1
    P[2] = 0.5 * ((α + β + 2) * x + (α - β))
    if N == 2
        return P
    end

    for n in 1:(N-2)
        a = (2n + α + β + 1) * (2n + α + β + 2) /
            (2 * (n + 1) * (n + α + β + 1))
        b = (β^2 - α^2) * (2n + α + β + 1) /
            (2 * (n + 1) * (n + α + β + 1) * (2n + α + β))
        c = (n + α) * (n + β) * (2n + α + β + 2) /
            ((n + 1) * (n + α + β + 1) * (2n + α + β))

        P[n+2] = (a * x - b) * P[n+1] - c * P[n]
    end

    return P
end

# L²-norm of shifted Jacobi polynomial basis on λ ∈ (0,1)
# with weight w(λ)=λ^α(1-λ)^β:
#
#   h_n = Γ(n+α+1) Γ(n+β+1)
#         /[(2n+α+β+1) n! Γ(n+α+β+1)].
#
function jacobi_norms_shifted(N::Int, α::Float64, β::Float64)
    inv_sqrt_h = zeros(Float64, N)
    for n in 0:(N-1)
        logh = -log(2n + α + β + 1) +
               loggamma(n + α + 1) + loggamma(n + β + 1) -
               loggamma(n + 1)     - loggamma(n + α + β + 1)
        inv_sqrt_h[n+1] = exp(-0.5 * logh)
    end
    return inv_sqrt_h
end

"""
    jacobi_phi_matrix(N, α, β, λ_nodes, inv_sqrt_h)

Build the matrix Φ with entries

  Φ[i,k] = φ_k(λ_i) = p_k(λ_i) * sqrt(w(λ_i)),

where p_k are orthonormal Jacobi polynomials on (0,1)
w.r.t. w(λ)=λ^α (1-λ)^β, and 0 ≤ k ≤ N-1.

Under x=2λ-1, w(λ) corresponds to standard Jacobi
weight (1-x)^β (1+x)^α, so we use J_n^{(β,α)}(x).
"""
function jacobi_phi_matrix(N::Int,
                           α::Float64,
                           β::Float64,
                           λ_nodes::Vector{Float64},
                           inv_sqrt_h::Vector{Float64})

    Q = length(λ_nodes)
    Φ = Matrix{Float64}(undef, Q, N)

    @inbounds for i in 1:Q
        λ = λ_nodes[i]
        # weight in λ-variable: w(λ) = λ^α (1-λ)^β
        logwλ  = α * log(λ) + β * log(1 - λ)
        sqrtwλ = exp(0.5 * logwλ)

        # map to x ∈ [-1,1]
        x = 2λ - 1

        # Jacobi polynomials with respect to (1-x)^β (1+x)^α
        Pvals = jacobiP_eval_all(N, β, α, x)   # J_0,...,J_{N-1}

        for k in 0:(N-1)
            Φ[i, k+1] = inv_sqrt_h[k+1] * Pvals[k+1] * sqrtwλ
        end
    end

    return Φ
end

"""
    fredholm_det_JUE(N, α, β, t0, inv_sqrt_h; N_quad=80)

Compute gap probability

  F_N(t0) = P(λ_max ≤ t0)

for JUE with parameters (α,β) (β=2 Jacobi ensemble on (0,1)),
via Nyström discretisation of the operator on (t0,1)
with kernel K_N(λ,μ).

We use Gauss–Legendre quadrature on (t0,1).
"""
function fredholm_det_JUE(N::Int,
                          α::Float64,
                          β::Float64,
                          t0::Float64,
                          inv_sqrt_h::Vector{Float64};
                          N_quad::Int=80)

    # Gauss–Legendre on (t0,1)
    z_leg, w_leg = gausslegendre(N_quad)      # on (-1,1)
    J = (1 - t0)/2
    λ = J .* (z_leg .+ 1.0) .+ t0             # nodes in (t0,1)
    dλ = J .* w_leg                           # Lebesgue weights

    # Φ(i,k) = φ_k(λ_i)
    Φ = jacobi_phi_matrix(N, α, β, λ, inv_sqrt_h)  # Q × N
    # K = Φ Φᵀ  (since K_N(λ_i,λ_j)=∑ φ_k(λ_i)φ_k(λ_j))
    K = Φ * transpose(Φ)                      # Q × Q

    # Nyström matrix A = sqrt(dλ) K sqrt(dλ)
    sqrt_dλ = sqrt.(dλ)
    A = copy(K)
    @inbounds for i in 1:N_quad
        A[i, :] .*= sqrt_dλ[i]
    end
    @inbounds for j in 1:N_quad
        A[:, j] .*= sqrt_dλ[j]
    end

    # F_N(t0) ≈ det(I - A)
    λA = eigvals(Symmetric(A))
    λA = clamp.(real.(λA), 0.0, 1.0 - 1e-12)
    logdet = sum(log.(1 .- λA))
    return exp(logdet)
end

"""
    FN_soft_grid(N, γ1, γ2, s_grid; N_quad=80)

For a given matrix size N and aspect ratios γ1,γ2 (so
n1 = γ1 N, n2 = γ2 N), compute the "soft-edge" CDF

  F_N^{soft}(s) = P( (logit λ_max - μ_C)/σ_C ≤ s )

deterministically on a grid s_grid, using:

- JUE parameters α = n2 - N, β = n1 - N
- Johnstone's μ_C, σ_C in the logit variable
- Fredholm determinant of JUE kernel for F_N(t)
"""
function FN_soft_grid(N::Int,
                      γ1::Float64,
                      γ2::Float64,
                      s_grid::Vector{Float64};
                      N_quad::Int=80)

    n1 = round(Int, γ1 * N)     # df of A
    n2 = round(Int, γ2 * N)     # df of B

    # JUE parameters for eigenvalues of (A+B)^{-1} B:
    # exponents λ^(α) (1-λ)^(β) with
    #   α = n2 - N, β = n1 - N
    α = float(n2 - N)
    β = float(n1 - N)

    # Johnstone's logit centering/scaling for complex double Wishart
    μC, σC = johnstone_complex_mu_sigma(N, n1, n2)

    # orthonormalisation constants for shifted Jacobi
    inv_sqrt_h = jacobi_norms_shifted(N, α, β)

    FNs = zeros(Float64, length(s_grid))
    for (k, s) in enumerate(s_grid)
        w  = μC + σC * s           # logit coordinate
        λs = logistic(w)           # back to (0,1)
        FNs[k] = fredholm_det_JUE(N, α, β, λs, inv_sqrt_h; N_quad=N_quad)
    end
    return FNs
end

###############################################################
# 4. Main soft-edge comparison: JUE vs Tracy–Widom F₂
###############################################################

function main()
    # 4.1 Tabulate F₂ on a grid
    s_min, s_max = -8.0, 4.0
    s_grid = collect(range(s_min, s_max; length=201))
    @info "Computing Tracy–Widom F₂ on $(length(s_grid)) points via Fredholm…"
    F2_grid = F2_fredholm_vec(s_grid; N=80)

    μ_F2, σ_F2 = tw_moments_from_cdf(s_grid, F2_grid)
    @info "Approximate TW F₂ moments" μ_F2 σ_F2

    # 4.2 JUE soft-edge for increasing N
    Ns = [100, 200, 400]
    γ1, γ2 = 2.0, 3.0           # same ratios as before

    for N in Ns
        @info "Computing deterministic JUE soft-edge CDF for N=$N…"
        FN_soft = FN_soft_grid(N, γ1, γ2, s_grid; N_quad=80)

        # sup norm difference between F_N^{soft} and F₂ on s-range
        supdiff = maximum(abs.(FN_soft .- F2_grid))

        println("=== Deterministic JUE soft-edge test (no Monte Carlo) ===")
        println("N = $N, n1 = $(round(Int,γ1*N)), n2 = $(round(Int,γ2*N))")
        println("  sup_s |F_N^{soft}(s) - F₂(s)| on [$s_min,$s_max] ≈ $(round(supdiff, digits=3))")
        println()
    end
end

main()
