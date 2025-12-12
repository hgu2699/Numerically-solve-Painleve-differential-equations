##############################
# PIV σ-form route for finite-n GUE (n = 20)
#
# - Fredholm Nyström for F_n(s) via Hermite kernel
# - 5-point finite-difference derivatives of log F_n
# - σ-form PIV IVP in (H, H', ℓ), where H = d/ds log F_n
# - Backward integration from anchor s0 to as far left as ODE remains regular
# - Comparison of F_sigma(s) vs Fredholm F_n(s)
#
# This is the "Route C" σ-form engine from the notes, cleaned up so it runs
# without domain errors. There is still a real pole around s ≈ 5.5 for the
# *branch* selected by the current anchoring, as explained in the notes.
##############################

using OrdinaryDiffEq
using FastGaussQuadrature
using LinearAlgebra
using Printf

# ============================================================
# 0. Hermite Nyström Fredholm determinant: F_n(s)
# ============================================================

"""
    hermite_phi_chain(n, x)

Return (ϕ_{n-1}(x), ϕ_n(x), sum_{k=0}^{n-1} ϕ_k(x)^2),

where ϕ_k are orthonormal Hermite functions (GUE normalisation).
"""
function hermite_phi_chain(n::Int, x::Float64)
    ϕ0 = π^(-0.25) * exp(-0.5*x*x)
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

"""
    K_hermite(n, s, t)

Finite-n Hermite kernel K_n(s,t).
"""
function K_hermite(n::Int, s::Float64, t::Float64)
    if s == t
        _, _, sumsq = hermite_phi_chain(n, s)
        return sumsq
    end
    φnm1_s, φn_s, _ = hermite_phi_chain(n, s)
    φnm1_t, φn_t, _ = hermite_phi_chain(n, t)
    return sqrt(n/2) * (φnm1_s*φn_t - φn_s*φnm1_t) / (s - t)
end

"""
    tail_len(n)

Heuristic tail length R_off for Nyström truncation (s, s+R_off].
"""
tail_len(n::Int) = n <= 50 ? 8.0 : (n <= 500 ? 10.0 : 12.0)

"""
    fredholm_cdf_gue_nystrom(n, s, R_off, Nquad)

Nyström approximation:
  F_n(s) = det(I - K_n |_{(s,∞)})

where we truncate to (s, s+R_off] and use Gauss–Legendre with Nquad points.
"""
function fredholm_cdf_gue_nystrom(n::Int, s::Float64,
                                  R_off::Float64,
                                  Nquad::Int)
    L = R_off
    z, w = gausslegendre(Nquad)          # nodes, weights on [-1,1]
    u  = (z .+ 1.0) .* (L/2)             # map to [0,L]
    dt = (L/2) .* w
    t  = s .+ u                          # quadrature nodes in (s,s+L]
    sq = sqrt.(dt)

    A = Matrix{Float64}(undef, Nquad, Nquad)
    @inbounds for j in 1:Nquad
        tj = t[j]
        for i in 1:Nquad
            A[i,j] = sq[i] * K_hermite(n, t[i], tj) * sq[j]
        end
    end

    λ = eigvals(Matrix(I - A))
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

"""
    logF_and_derivs_5pt(n, s0; h, R_off, Nquad)

Compute ℓ(s0)=log F_n(s0), ℓ'(s0), ℓ''(s0), ℓ'''(s0) using
5-point central finite differences in s.

This is used to anchor the σ-form IVP.
"""
function logF_and_derivs_5pt(n::Int, s0::Float64;
                             h::Float64 = 0.02,
                             R_off::Float64 = tail_len(n),
                             Nquad::Int = 260)

    s_vals = (s0 - 2h, s0 - h, s0, s0 + h, s0 + 2h)
    F_vals = [fredholm_cdf_gue_nystrom(n, s, R_off, Nquad) for s in s_vals]
    ℓm2, ℓm1, ℓ0, ℓp1, ℓp2 = log.(F_vals)

    # standard 5-point central formulas
    ℓ1 = (ℓm2 - 8ℓm1 + 8ℓp1 - ℓp2) / (12h)
    ℓ2 = (-ℓp2 + 16ℓp1 - 30ℓ0 + 16ℓm1 - ℓm2) / (12h^2)
    ℓ3 = (ℓp2 - 2ℓp1 + 2ℓm1 - ℓm2) / (2h^3)

    return ℓ0, ℓ1, ℓ2, ℓ3
end

# ============================================================
# 1. σ-form PIV IVP: state Y = (H, H', ℓ)
# ============================================================

"""
Parameters for σ-form PIV IVP.

- n: matrix size (GUE)
- branch_sign: +1 or -1, picks the local H'' branch at the anchor.
"""
struct SigmaPIVParams
    n::Int
    branch_sign::Float64
end

"""
    sigma_piv_rhs!(dY, Y, p, s)

Right-hand side for σ-form PIV IVP.

Y = [H, Hp, ℓ] where:
  H  = H(s)  = d/ds log F_n(s)
  Hp = H'(s)
  ℓ  = log F_n(s)

Equations:
  H'  = Hp
  Hp' = H'' from σ-form of PIV:
         (H'')^2 - 4 (s H' - H)^2 + 4 H'^2 (H' + 2n) = 0
       ⇒ H'' = branch_sign * sqrt(disc)
  ℓ'  = H
"""
function sigma_piv_rhs!(dY, Y, p::SigmaPIVParams, s)
    H   = Y[1]
    Hp  = Y[2]
    n   = p.n
    sgn = p.branch_sign

    term1 = 4.0 * (s*Hp - H)^2
    term2 = 4.0 * (Hp^2) * (Hp + 2.0*n)
    disc  = term1 - term2

    # numerical guard: allow tiny negative disc from roundoff
    if disc < 0.0
        if disc > -1e-12 * (abs(term1) + abs(term2) + 1.0)
            disc = 0.0
        else
            # let the solver crash if we truly go complex
            error("σ-form discriminant became negative at s = $s: disc = $disc")
        end
    end

    Hpp = sgn * sqrt(disc)

    dY[1] = Hp     # H'
    dY[2] = Hpp    # H''
    dY[3] = H      # ℓ' = H

    return nothing
end

# ============================================================
# 2. Solve σ-form IVP from a finite anchor
# ============================================================

"""
    solve_sigma_from_anchor(n, s_min;
                            s0, h, R_off, Nquad, n_save)

- n: GUE matrix size (here we'll use n=20)
- s0: anchor point near the edge (e.g. 6.4)
- s_min: left end of interval we *want* to reach (e.g. 5.0)
- h: FD step for derivatives
- R_off, Nquad: Nyström tail and quad points
- n_save: number of sample points between s0 and s_left

Uses Fredholm Nyström at s0 to compute:
  ℓ0 = log F_n(s0),
  H0 = ℓ'(s0),
  Hp0 = ℓ''(s0),
  Hpp_est ≈ ℓ'''(s0).

Then chooses branch_sign based on H'' and integrates σ-PIV
backwards from s0 with a stiff solver. Returns:

  s_all   -- vector of s values in [s_left, s0]
  F_sigma -- exp(ℓ(s)) from σ-form IVP
  F_fred  -- Fredholm determinant at same s
"""
function solve_sigma_from_anchor(n::Int, s_min::Float64;
                                 s0::Float64    = 6.4,
                                 h::Float64     = 0.02,
                                 R_off::Float64 = tail_len(n),
                                 Nquad::Int     = 260,
                                 n_save::Int    = 80)

    # 1) Anchor from Fredholm
    ℓ0, H0, Hp0, Hpp_est =
        logF_and_derivs_5pt(n, s0; h=h, R_off=R_off, Nquad=Nquad)

    @printf("\n--- σ-form anchor at s0 = %.4f ---\n", s0)
    @printf("log F_n(s0) ≈ %.12e\n", ℓ0)
    @printf("H(s0)       ≈ %.12e\n", H0)
    @printf("H'(s0)      ≈ %.12e\n", Hp0)
    @printf("H''(s0)     ≈ %.12e (finite differences)\n\n", Hpp_est)

    # 2) Choose branch sign so σ-form H'' matches FD H'' in sign
    #    If H'' is tiny, just pick +1.
    branch_sign = abs(Hpp_est) < 1e-10 ? 1.0 : sign(Hpp_est)
    params      = SigmaPIVParams(n, branch_sign)

    # 3) Solve IVP: Y = [H, H', ℓ]
    Y0    = [H0, Hp0, ℓ0]
    tspan = (s0, s_min)   # backwards integration

    prob = ODEProblem(sigma_piv_rhs!, Y0, tspan, params)

    # BDF method (QNDF) with finite-difference Jacobian (no AD shenanigans)
    sol = solve(prob,
                QNDF();
                reltol  = 1e-11,
                abstol  = 1e-13,
                maxiters = 10^7)

    s_left = minimum(sol.t)
    s_right = maximum(sol.t)

    @printf("--- σ-form IVP done ---\n")
    @printf("Solver retcode: %s\n", string(sol.retcode))
    @printf("Solution covers s ∈ [%.6f, %.6f]\n\n", s_left, s_right)

    # 4) Sample only where the solution actually exists:
    s_low = max(s_min, s_left)
    s_all = collect(LinRange(s_low, s0, n_save))

    ℓ_sigma = similar(s_all)
    F_sigma = similar(s_all)
    F_fred  = similar(s_all)

    for (i, s) in pairs(s_all)
        Y = sol(s)               # interpolation
        ℓ_sigma[i] = Y[3]
        F_sigma[i] = exp(ℓ_sigma[i])
        F_fred[i]  = fredholm_cdf_gue_nystrom(n, s, R_off, Nquad)
    end

    return s_all, F_sigma, F_fred
end

# ============================================================
# 3. Driver: compare σ-form vs Fredholm
# ============================================================

function main()
    n_val = 20
    s0    = 6.40      # anchor just right of the edge
    s_min = 5.00      # we would like to go this far, but ODE may stop earlier
    h     = 0.02
    R_off = tail_len(n_val)
    Nquad = 260
    n_save = 80

    s_all, F_sigma, F_fred =
        solve_sigma_from_anchor(n_val, s_min;
                                s0     = s0,
                                h      = h,
                                R_off  = R_off,
                                Nquad  = Nquad,
                                n_save = n_save)

    diffs = abs.(F_sigma .- F_fred)
    maxdiff, idx = findmax(diffs)

    println("==============================================")
    println("σ-form PIV (Hamiltonian) vs Fredholm Nyström")
    @printf("n = %d, s ∈ [%.3f, %.3f]\n", n_val, minimum(s_all), maximum(s_all))
    println("==============================================")
    @printf("%8s  %15s  %15s  %10s\n", "s", "F_sigma", "F_Fredholm", "diff")
    @printf("%s\n", "-"^60)
    for j in 1:3:length(s_all)
        @printf("%8.3f  %15.8e  %15.8e  %10.3e\n",
                s_all[j], F_sigma[j], F_fred[j], diffs[j])
    end
    println()

    @printf("Max |F_sigma - F_Fred| ≈ %.3e at s ≈ %.6f\n",
            maxdiff, s_all[idx])
end

main()
