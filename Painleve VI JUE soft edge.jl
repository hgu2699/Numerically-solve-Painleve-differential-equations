###############################################################
# jue_soft_edge_F2.jl
#
# Goal: Numerically check that the normalized largest eigenvalue
#       of the Jacobi unitary ensemble (realized as a double
#       Wishart / MANOVA model) converges in law to Tracy–Widom F₂.
#
# - F₂ is computed as Fredholm determinant of Airy kernel (Nyström)
# - JUE is simulated by eigenvalues of (A+B)^{-1} B with
#   complex Wisharts A,B
#
# No RandomMatrixDistributions / ApproxFun used.
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

using FastGaussQuadrature
using SpecialFunctions     # airyai, airyaiprime
using LinearAlgebra
using Random
using Statistics

const Ai  = airyai
const Aip = airyaiprime

###########################################
# 1. Tracy–Widom F₂ via Airy Fredholm det #
###########################################

# Airy kernel
# K(s,t) = (Ai(s)Ai'(t) - Ai'(s)Ai(t))/(s - t),
# with diagonal K(s,s) = Ai'(s)^2 - s*Ai(s)^2
function airy_kernel(s::Float64, t::Float64)
    if s == t
        return Aip(s)^2 - s*Ai(s)^2
    else
        return (Ai(s)*Aip(t) - Aip(s)*Ai(t)) / (s - t)
    end
end

"""
    F2_fredholm(x; N=80)

Tracy–Widom F₂ CDF evaluated at x via Nyström method on (x,∞):

- Map z∈(0,1) → t = x + z/(1-z), dt = dz/(1-z)^2
- Use Gauss–Legendre on z∈(0,1)
"""
function F2_fredholm(x::Float64; N::Int=80)
    # Gauss–Legendre nodes/weights on (-1,1)
    z, w = gausslegendre(N)
    # Map to (0,1)
    z  = (z .+ 1.0) ./ 2.0
    w  = w ./ 2.0

    # Map to (x,∞)
    t  = x .+ z ./ (1 .- z)
    dt = w ./ (1 .- z).^2

    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)

    @inbounds for i in 1:N
        ti = t[i]
        for j in 1:N
            tj = t[j]
            Kij = (i == j) ? (Aip(ti)^2 - ti*Ai(ti)^2) :
                             ((Ai(ti)*Aip(tj) - Aip(ti)*Ai(tj)) / (ti - tj))
            A[i,j] = sq[i] * Kij * sq[j]
        end
    end

    # B = I - A, take determinant via eigenvalues
    B = Symmetric(Matrix(I, N, N) - A)
    λ = eigvals(B)
    λ = clamp.(real.(λ), eps(), 1.0)  # numerical safety
    return exp(sum(log, λ))
end

F2_fredholm_vec(xs::AbstractVector{<:Real}; N::Int=80) =
    [F2_fredholm(Float64(x); N=N) for x in xs]

# Simple linear interpolation y(x) on a grid
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

# Approximate mean and std of F₂ from a tabulated CDF
function tw_moments_from_cdf(x::Vector{Float64}, F::Vector{Float64})
    @assert length(x) == length(F)
    # probabilities in each interval
    p    = diff(F)
    xmid = @. (x[1:end-1] + x[2:end]) / 2
    μ    = sum(p .* xmid)
    σ2   = sum(p .* (xmid .- μ).^2)
    return μ, sqrt(σ2)
end

##########################################
# 2. Double Wishart simulator for the JUE #
##########################################

"""
    sample_JUE_lmax(N, γ1, γ2; M=2000)

Simulate M samples of the largest eigenvalue of (A+B)^{-1} B where

- A = X Xʰ, X ∈ ℂ^{N×n1},  n1 = round(γ1*N)
- B = Y Yʰ, Y ∈ ℂ^{N×n2},  n2 = round(γ2*N)

The eigenvalues lie in (0,1); this is the Jacobi unitary ensemble
(double Wishart / MANOVA model).
"""
function sample_JUE_lmax(N::Int, γ1::Float64, γ2::Float64; M::Int=2000)
    n1 = round(Int, γ1 * N)
    n2 = round(Int, γ2 * N)

    λmax = zeros(Float64, M)
    for m in 1:M
        X = randn(ComplexF64, N, n1)
        Y = randn(ComplexF64, N, n2)
        A = X * X'   # Wishart(N, n1)
        B = Y * Y'   # Wishart(N, n2)

        vals = eigen(Hermitian(B), Hermitian(A + B)).values
        λmax[m] = maximum(real.(vals))
    end
    return λmax
end

# Empirical CDF of samples on a given grid
function empirical_cdf_grid(samples::Vector{Float64}, grid::Vector{Float64})
    M = length(samples)
    sorted = sort(samples)
    F = similar(grid, Float64)
    j = 1
    for (k, z) in enumerate(grid)
        while j <= M && sorted[j] <= z
            j += 1
        end
        F[k] = (j - 1) / M
    end
    return F
end

##############################
# 3. Main comparison routine #
##############################

function main()
    # 3.1 Tabulate F₂ on a grid and standardize it
    s_min, s_max = -8.0, 4.0
    s_grid = collect(range(s_min, s_max; length=401))
    @info "Computing Tracy–Widom F₂ on $(length(s_grid)) points via Fredholm…"
    F2_grid = F2_fredholm_vec(s_grid; N=80)

    μ_F2, σ_F2 = tw_moments_from_cdf(s_grid, F2_grid)
    @info "Approximate TW F₂ moments" μ_F2 σ_F2

    # Build standardized F₂ CDF on a z-grid
    z_grid = collect(range(-4.0, 4.0; length=401))
    s_from_z = @. μ_F2 + σ_F2 * z_grid
    F2_std_grid = interp1(s_grid, F2_grid, s_from_z)

    # 3.2 JUE soft-edge test for increasing N
    Ns = [100, 200, 400]
    γ1, γ2 = 2.0, 3.0         # aspect ratios n1/N, n2/N
    M = 5000                  # Monte Carlo samples per N

    for N in Ns
        @info "Simulating JUE largest eigenvalues for N=$N, M=$M…"
        λ = sample_JUE_lmax(N, γ1, γ2; M=M)

        # Normalise: mean 0, var 1
        μλ = mean(λ)
        σλ = std(λ)
        z_samples = (λ .- μλ) ./ σλ

        # Empirical CDF of normalised λ_max on z_grid
        F_emp = empirical_cdf_grid(z_samples, z_grid)

        # Compare to standardised F₂
        supdiff = maximum(abs.(F_emp .- F2_std_grid))

        # Check N^{-2/3}-scaling of fluctuations
        c_est = σλ * N^(2/3)

        println("=== JUE soft-edge test (double Wishart) ===")
        println("N = $N, n1 ≈ $(round(Int,γ1*N)), n2 ≈ $(round(Int,γ2*N)), M = $M")
        println("  mean(λ_max) ≈ $(round(μλ, digits=5))")
        println("  std(λ_max)  ≈ $(round(σλ, digits=5))")
        println("  σ_N * N^(2/3) ≈ $(round(c_est, digits=5))   (should stabilise with N)")
        println("  sup_z |F_N(z) - F₂_std(z)| on [-4,4] ≈ $(round(supdiff, digits=3))")
        println()
    end
end

# Run when executed as script
main()
