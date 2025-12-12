#############################
# JUE hard-edge → Bessel
#############################

import Pkg
for p in ["FastGaussQuadrature", "SpecialFunctions", "LinearAlgebra",
          "Plots", "Printf"]
    Base.find_package(p) === nothing &&
        (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature
using SpecialFunctions          # jacobi, besselj, gamma
using LinearAlgebra
using Plots
using Printf

gr()   # Plots backend

# -------------------------------------------------------
# 1. Bessel kernel and gap probability (validated style)
# -------------------------------------------------------

# Bessel kernel K_α(x,y), x,y > 0
function K_bessel(α::Float64, x::Float64, y::Float64)
    if x == y
        t = sqrt(x)
        Jα   = besselj(α,   t)
        Jαp1 = besselj(α+1, t)
        Jαm1 = besselj(α-1, t)
        return 0.25 * (Jα^2 - Jαp1*Jαm1)
    else
        sx, sy = sqrt(x), sqrt(y)
        Jαx = besselj(α, sx)
        Jαy = besselj(α, sy)
        Jαpx = 0.5*(besselj(α-1, sx) - besselj(α+1, sx))
        Jαpy = 0.5*(besselj(α-1, sy) - besselj(α+1, sy))
        return (Jαx*sy*Jαpy - Jαy*sx*Jαpx) / (2*(x-y))
    end
end

# Fredholm determinant on (0,s) for Bessel kernel
function F_Bessel_gap(s::Float64; α::Float64=0.0, nquad::Int=120)
    # Gauss-Legendre on (0,s)
    z, w = gausslegendre(nquad)              # (-1,1)
    xs = (s/2) .* (z .+ 1.0)                 # (0,s)
    ws = (s/2) .* w

    sqw = sqrt.(ws)
    A = Matrix{Float64}(undef, nquad, nquad)

    @inbounds for j in 1:nquad
        y = xs[j]
        for i in 1:nquad
            x = xs[i]
            A[i,j] = sqw[i] * K_bessel(α, x, y) * sqw[j]
        end
    end

    λ = eigvals(I - A)
    λ = clamp.(real.(λ), eps(), 1.0)         # numerical safety
    return exp(sum(log, λ))
end

# -------------------------------------------------------
# 2. JUE kernel via orthonormal Jacobi polynomials
#    weight: w(x) = (1-x)^a (1+x)^b on (-1,1)
# -------------------------------------------------------

# Norm h_n of standard Jacobi P_n^{(a,b)} w.r.t. w(x) dx
function jacobi_norm(n::Int, a::Float64, b::Float64)
    num = 2.0^(a+b+1.0) * gamma(n+a+1.0) * gamma(n+b+1.0)
    den = (2n + a + b + 1.0) * gamma(n+1.0) * gamma(n+a+b+1.0)
    return num / den
end

# Row of orthonormal functions φ_k(x) for k=0,...,N-1
function jacobi_phi_row(N::Int, a::Float64, b::Float64, x::Float64)
    w0 = (1.0 - x)^a * (1.0 + x)^b
    row = Vector{Float64}(undef, N)
    @inbounds for k in 0:(N-1)
        Pk = jacobi(k, a, b, x)              # SpecialFunctions.jacobi
        hk = jacobi_norm(k, a, b)
        row[k+1] = Pk * sqrt(w0 / hk)
    end
    return row
end

# Φ_{k,j} = φ_k(x_j)
function jue_phi_matrix(N::Int, a::Float64, b::Float64, xs::Vector{Float64})
    M = length(xs)
    Φ = Matrix{Float64}(undef, N, M)
    @inbounds for j in 1:M
        Φ[:,j] = jacobi_phi_row(N, a, b, xs[j])
    end
    return Φ
end

# Fredholm determinant for JUE gap on (x_lo, x_hi)
function F_JUE_gap_interval(N::Int, a::Float64, b::Float64,
                            x_lo::Float64, x_hi::Float64; nquad::Int=120)
    # Gauss-Legendre on (x_lo, x_hi)
    z, w = gausslegendre(nquad)
    xs = ((x_hi - x_lo)/2.0) .* (z .+ 1.0) .+ x_lo
    ws = ((x_hi - x_lo)/2.0) .* w

    Φ = jue_phi_matrix(N, a, b, xs)
    K = transpose(Φ) * Φ                     # K_{ij} = sum_k φ_k(x_i)φ_k(x_j)

    sqw = sqrt.(ws)
    A = Matrix{Float64}(undef, nquad, nquad)
    @inbounds for j in 1:nquad
        for i in 1:nquad
            A[i,j] = sqw[i] * K[i,j] * sqw[j]
        end
    end

    λ = eigvals(I - A)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

# Hard-edge scaling near x = -1:
# x = -1 + s / (2 N^2)   (theoretical scaling up to lower-order corrections)
function F_JUE_gap_scaled(N::Int, a::Float64, b::Float64, s::Float64;
                          nquad_jue::Int=120)
    t = s / (2.0 * N^2)            # length of interval near -1
    x_lo = -1.0
    x_hi = -1.0 + t
    return F_JUE_gap_interval(N, a, b, x_lo, x_hi; nquad=nquad_jue)
end

# -------------------------------------------------------
# 3. Test hard-edge scaling JUE → Bessel for N = 20, 40, 80
# -------------------------------------------------------

function test_jue_hard_edge(; Ns = [20,40,80],
                            a::Float64 = 0.0,
                            b::Float64 = 0.0,
                            s_min::Float64 = 1.0,
                            s_max::Float64 = 10.0,
                            ns::Int = 10,
                            nquad_jue::Int = 120,
                            nquad_bes::Int = 120)

    sgrid = collect(range(s_min, s_max; length=ns))
    α = b    # Bessel index at the left endpoint x=-1

    for N in Ns
        @printf("\n=== JUE hard-edge scaling test (N = %d, a = %.1f, b = %.1f) ===\n",
                N, a, b)
        # Bessel gap on (0,s)
        F_Bes = [F_Bessel_gap(s; α=α, nquad=nquad_bes) for s in sgrid]

        # JUE gap on (-1, -1 + s/(2N^2))
        F_JUE = [F_JUE_gap_scaled(N, a, b, s; nquad_jue=nquad_jue) for s in sgrid]

        # Print max error
        err = abs.(F_JUE .- F_Bes)
        @printf("Max |F_JUE - F_Bes| over s ∈ [%.1f, %.1f] ≈ %.6e\n",
                s_min, s_max, maximum(err))

        # Plots
        plt1 = plot(sgrid, F_Bes, lw=2, label="Bessel gap (α=$(α))",
                    xlabel="s", ylabel="Gap CDF",
                    title="JUE hard-edge scaling vs Bessel (N=$(N), a=$(a))")
        plot!(plt1, sgrid, F_JUE, lw=2, ls=:dash,
              label="JUE gap, N=$(N)")
        savefig(plt1, "jue_hard_edge_vs_bessel_N$(N)_a$(Int(round(a))).png")
        display(plt1)

        plt2 = plot(sgrid, err, lw=2, label="abs error",
                    xlabel="s", ylabel="|F_JUE - F_Bes|",
                    title="JUE→Bessel hard-edge error (N=$(N), a=$(a))")
        savefig(plt2, "jue_hard_edge_error_N$(N)_a$(Int(round(a))).png")
        display(plt2)
    end
end

# Run the test when file is executed
test_jue_hard_edge()
