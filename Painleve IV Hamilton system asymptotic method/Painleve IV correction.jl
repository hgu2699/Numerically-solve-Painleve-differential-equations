##############################
# Painlevé IV via Hamiltonian
# for finite-n GUE (n = 20)
#
# Route B: Fredholm-anchored IVP
##############################

import Pkg
for p in ["OrdinaryDiffEq","FastGaussQuadrature","LinearAlgebra","Printf"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using OrdinaryDiffEq
using FastGaussQuadrature
using LinearAlgebra
using Printf

# ============================================================
# 0. ASYMPTOTIC COEFFICIENTS FOR n = 20 (q,p at large s)
# ============================================================

# q(s) ~ C_n s^(2n) e^{-s^2} * hat_q(1/s),
# p(s) ~ hat_p(1/s), with these 1/s coefficients (A = 1).

const Q_COEFFS_20 = [
    1.0, 0.0, -190.0, 0.0, 16292.5, 0.0, -835762.5, 0.0, 2.86657453125e7,
    0.0, -6.961465575e8, 0.0, 1.2367113834375e10, 0.0, -1.63778990090625e11,
    0.0, 1.6329840592939453e12, 0.0, -1.229944372883789e13, 0.0,
    6.9825937693523734e13, 0.0, -2.965911072963706e14, 0.0,
    9.304457170258372e14, 0.0, -2.114351544225575e15, 0.0,
    3.3849280342596835e15, 0.0, -3.669930094858188e15, 0.0,
    2.5444441662901055e15, 0.0, -1.0334760366936674e15, 0.0,
    2.1258209615534297e14, 0.0, -1.6352468935026383e13, 0.0,
    1.8040022082762219e28
]

const P_COEFFS_20 = [
    0.0, 10.0, 0.0, 95.0, 0.0, 1757.5, 0.0, 39781.25, 0.0, 990315.625,
    0.0, 2.59971359375e7, 0.0, 7.0494953359375e8, 0.0, 1.9519781805078125e10,
    0.0, 5.4810960368066406e11, 0.0, 1.5539038761020996e13, 0.0,
    4.434910518497595e14, 0.0, 1.2717326406832328e16, 0.0,
    3.659040934415078e17, 0.0, 1.0553242233288081e19, 0.0,
    3.049010990778647e20, 0.0, 8.820181583012918e21, 0.0,
    2.5538268124701118e23, 0.0, 7.399331842382315e24, 0.0,
    2.144877057943033e26, 0.0, 6.2196270695401e27, 0.0
]

# ============================================================
# 1. LARGE-s ASYMPTOTICS (for branch selection)
# ============================================================

"""
    get_asymptotic_qp_unit(s_val, n_val)

Return (q0, p) where:
  - q0(s) is the large-s asymptotic q(s) with amplitude A = 1:
        q0(s) = C_n s^(2n) e^{-s^2} * hat_q(1/s),
  - p(s) is the asymptotic p(s) = hat_p(1/s).

We only use this for picking the correct root when anchoring.
"""
function get_asymptotic_qp_unit(s_val, n_val::Int)
    @assert n_val == 20 "This asymptotic coefficient set is for n = 20."

    C_n = 2.0^(n_val - 1) / (sqrt(pi) * float(factorial(big(n_val) - 1)))

    # hat_q(1/s)
    s_inv = 1.0 / s_val
    hat_q = 0.0
    pow = 1.0
    for k in 0:(length(Q_COEFFS_20) - 1)
        hat_q += Q_COEFFS_20[k+1] * pow
        pow *= s_inv
    end

    # p(s) ~ hat_p(1/s)
    p_val = 0.0
    s_inv = 1.0 / s_val
    pow = 1.0
    for k in 0:(length(P_COEFFS_20) - 1)
        p_val += P_COEFFS_20[k+1] * pow
        pow *= s_inv
    end

    q0_val = C_n * s_val^(2*n_val) * exp(-s_val*s_val) * hat_q
    return float(q0_val), float(p_val)
end

# ============================================================
# 2. HAMILTONIAN AND FREDHOLM SIDE
# ============================================================

"""
    H_hamiltonian(q, p, s, n_val)

Hamiltonian for the PIV system:
  H(q,p,s) = (2p - q - 2s) p q + n_val * q

Along PIV solutions:
  dH/ds = -2 p q
"""
function H_hamiltonian(q, p, s, n_val)
    return (2.0*p - q - 2.0*s) * p * q + n_val * q
end

# ---------- Fredholm Nyström side ----------

"""
    hermite_phi_chain(n, x)

Compute φ_{n-1}(x), φ_n(x), and sum_{k=0}^{n-1} φ_k(x)^2
for the normalized Hermite functions associated with the
n×n GUE Hermite kernel.
"""
function hermite_phi_chain(n::Int, x::Float64)
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

"""
    K_hermite(n, s, t)

Finite-n Hermite kernel for GUE.
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

Heuristic tail length R_offset for the [s, s+R] interval.
"""
tail_len(n::Int) = n ≤ 50 ? 8.0 : (n ≤ 500 ? 10.0 : 12.0)

"""
    fredholm_cdf_gue_nystrom(n, s, R_offset, Nquad)

Compute F_n(s) = P(λ_max ≤ s) for n×n GUE via Nyström
discretization of the Hermite kernel on [s, s+R_offset]
with Nquad Gauss–Legendre nodes.
"""
function fredholm_cdf_gue_nystrom(n::Int, s::Float64,
                                  R_offset::Float64,
                                  Nquad::Int)
    L = R_offset
    z, w = gausslegendre(Nquad)
    u  = (z .+ 1.0) .* (L/2)
    dt = (L/2) .* w
    t  = s .+ u
    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, Nquad, Nquad)
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
    fredholm_logF_and_derivs(n, s0, R_off, Nquad; h = 0.02)

Return (ℓ0, H_F, Hp_F) where
  ℓ0   ≈ log F_n(s0),
  H_F  ≈ d/ds log F_n(s0),
  Hp_F ≈ d^2/ds^2 log F_n(s0),

using 3-point central finite differences with step h.
"""
function fredholm_logF_and_derivs(n::Int, s0::Float64,
                                  R_off::Float64,
                                  Nquad::Int;
                                  h::Float64 = 0.02)
    Fm = fredholm_cdf_gue_nystrom(n, s0 - h, R_off, Nquad)
    F0 = fredholm_cdf_gue_nystrom(n, s0,     R_off, Nquad)
    Fp = fredholm_cdf_gue_nystrom(n, s0 + h, R_off, Nquad)

    ℓm, ℓ0, ℓp = log(Fm), log(F0), log(Fp)

    HF  = (ℓp - ℓm) / (2h)
    HpF = (ℓp - 2ℓ0 + ℓm) / (h*h)

    return ℓ0, HF, HpF
end

# ============================================================
# 3. ANCHOR (q0, p0) FROM FREDHOLM H and H'
# ============================================================

"""
    anchor_qp_from_fredholm(n_val, s0, R_off, Nquad; h = 0.02)

At anchor point s0:

- Compute ℓ0, H_F, Hp_F from Fredholm (log F, 1st and 2nd derivatives).
- Solve the system:
    H(q0,p0,s0)     = H_F,
    dH/ds(q0,p0,s0) = Hp_F = -2 q0 p0,
  for (q0,p0).

Use the large-s asymptotic (q_asym,p_asym) at s0 to pick the correct root.

Returns (q0, p0, ℓ0).
"""
function anchor_qp_from_fredholm(n_val::Int,
                                 s0::Float64,
                                 R_off::Float64,
                                 Nquad::Int;
                                 h::Float64 = 0.02)

    # Fredholm: logF and its derivatives
    ℓ0, HF, HpF = fredholm_logF_and_derivs(n_val, s0, R_off, Nquad; h=h)

    # Quadratic in q0:
    #  (HpF/2 + n) q0^2 + (HpF*s0 - HF) q0 + (HpF^2 / 2) = 0
    A = HpF/2 + n_val
    B = HpF*s0 - HF
    C = (HpF^2) / 2.0

    disc = B*B - 4.0*A*C
    if disc < 0
        error("anchor_qp_from_fredholm: negative discriminant; adjust s0 or h.")
    end
    sqrt_disc = sqrt(disc)
    q1 = (-B + sqrt_disc) / (2.0*A)
    q2 = (-B - sqrt_disc) / (2.0*A)

    # Corresponding p values from -2 q p = HpF
    p1 = HpF ≈ 0 ? 0.0 : (-HpF) / (2.0*q1)
    p2 = HpF ≈ 0 ? 0.0 : (-HpF) / (2.0*q2)

    # Asymptotic branch at s0 (for root selection)
    q_asym, p_asym = get_asymptotic_qp_unit(s0, n_val)

    d1 = (q1 - q_asym)^2 + (p1 - p_asym)^2
    d2 = (q2 - q_asym)^2 + (p2 - p_asym)^2

    if isfinite(d1) && isfinite(d2)
        if d1 <= d2
            q0, p0 = q1, p1
        else
            q0, p0 = q2, p2
        end
    elseif isfinite(d1)
        q0, p0 = q1, p1
    elseif isfinite(d2)
        q0, p0 = q2, p2
    else
        error("anchor_qp_from_fredholm: both roots are non-finite.")
    end

    @printf("\n--- Anchor from Fredholm at s0 = %.4f ---\n", s0)
    @printf("log F_n(s0)   ≈ %.12e\n", ℓ0)
    @printf("H_F(s0)       ≈ %.12e\n", HF)
    @printf("H'_F(s0)      ≈ %.12e\n", HpF)
    @printf("Asym q,p      ≈ (%.6e, %.6e)\n", q_asym, p_asym)
    @printf("Chosen q0,p0  ≈ (%.6e, %.6e)\n", q0, p0)
    @printf("Check H(q0,p0): %.12e\n", H_hamiltonian(q0, p0, s0, n_val))

    return q0, p0, ℓ0
end

# ============================================================
# 4. IVP: PIV HAMILTONIAN SYSTEM (q,p,ℓ)
# ============================================================

"""
    piv_ham_system!(dY, Y, p, s)

IVP system (SciML convention: f!(du, u, p, t)):

  dq/ds = q(4p - q - 2s)
  dp/ds = 2p(q + s - p) - n_val
  dℓ/ds = H(q,p,s) = (2p - q - 2s) p q + n_val q

Here:
  - parameter p = n_val (integer, e.g. 20),
  - time s is the spectral variable.
"""
function piv_ham_system!(dY, Y, p, s)
    n_val = p

    q  = Y[1]
    pY = Y[2]
    ℓ  = Y[3]  # not explicitly used, but kept for clarity

    dqds = q * (4.0*pY - q - 2.0*s)
    dpds = 2.0*pY * (q + s - pY) - n_val
    dℓds = H_hamiltonian(q, pY, s, n_val)

    dY[1] = dqds
    dY[2] = dpds
    dY[3] = dℓds
end

"""
    solve_piv_hamiltonian_ivp(n_val, s_target, s_anchor,
                              R_off, Nquad;
                              n_save = 60)

- Anchor (q0,p0,ℓ0) at s_anchor from Fredholm H,H' and asymptotics.
- Solve IVP for (q,p,ℓ) backward from s_anchor down to s_target.
- Sample solution at n_save points between s_target and s_anchor.

Returns (s_all, ℓ_all, success_flag).
"""
function solve_piv_hamiltonian_ivp(n_val::Int,
                                   s_target::Float64,
                                   s_anchor::Float64,
                                   R_off::Float64,
                                   Nquad::Int;
                                   n_save::Int = 60)

    # 0) Anchor at s_anchor
    q0, p0, ℓ0 = anchor_qp_from_fredholm(n_val, s_anchor, R_off, Nquad; h=0.02)
    Y0 = [q0, p0, ℓ0]

    println()
    println("--- Solving PIV Hamiltonian IVP (anchored) ---")
    @printf("n = %d, s_target = %.3f, s_anchor = %.3f\n\n", n_val, s_target, s_anchor)

    # 1) Solve IVP from s_anchor down to s_target
    tspan = (s_anchor, s_target)  # backward integration in s
    prob = ODEProblem(piv_ham_system!, Y0, tspan, n_val)
    sol = solve(prob, Rodas5(), reltol=1e-11, abstol=1e-13)

    # 2) Sample solution at a regular grid for comparison
    s_all = collect(LinRange(s_target, s_anchor, n_save))
    ℓ_all = zeros(Float64, n_save)

    for (i, s) in pairs(s_all)
        if s < minimum(sol.t) || s > maximum(sol.t)
            error("Requested s=$s outside solution interval.")
        end
        Y = sol(s)
        ℓ_all[i] = Y[3]
    end

    return s_all, ℓ_all, true
end

# ============================================================
# 5. MAIN DRIVER: PIV vs FREDHOLM (IVP, ANCHORED)
# ============================================================

function main()
    n_val     = 20
    s_anchor  = 6.40        # anchor point
    s_target  = 5.00        # left endpoint
    R_off     = tail_len(n_val)
    Nquad     = 260

    # 1) Solve Hamiltonian PIV IVP with Fredholm-anchored (q0,p0,ℓ0)
    s_all, ℓ_all, success =
        solve_piv_hamiltonian_ivp(n_val,
                                  s_target,
                                  s_anchor,
                                  R_off,
                                  Nquad;
                                  n_save = 60)

    if !success
        println("WARNING: PIV IVP solver signaled failure.")
    end

    # Convert to F_PIV(s)
    F_piv_all = exp.(ℓ_all)

    # 2) Compute Fredholm reference at the same s-points
    F_fred_all = [fredholm_cdf_gue_nystrom(n_val, s, R_off, Nquad) for s in s_all]

    # 3) s_all is ascending (s_target -> s_anchor)
    diffs = abs.(F_piv_all .- F_fred_all)
    maxdiff, idx = findmax(diffs)

    println("==============================================")
    println("Painlevé IV Hamiltonian IVP (anchored) vs Fredholm")
    println("n = $n_val, s ∈ [$(minimum(s_all)), $(maximum(s_all))]")
    println("Anchored at s_anchor using Fredholm H and H'.")
    println("==============================================")
    @printf("%8s  %15s  %15s  %10s\n", "s", "F_PIV", "F_Fredholm", "diff")
    @printf("%s\n", "-"^60)
    for j in 1:3:length(s_all)
        @printf("%8.3f  %15.8e  %15.8e  %10.3e\n",
                s_all[j], F_piv_all[j], F_fred_all[j], diffs[j])
    end
    println()

    @printf("Max |F_PIV - F_Fred| ≈ %.3e at s ≈ %.6f\n",
            maxdiff, s_all[idx])
    @printf("  F_PIV(s)  ≈ %.12e\n", F_piv_all[idx])
    @printf("  F_Fred(s) ≈ %.12e\n", F_fred_all[idx])
end

main()
