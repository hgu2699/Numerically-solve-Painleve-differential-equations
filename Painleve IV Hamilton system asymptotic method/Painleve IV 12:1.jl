##############################
# Painlevé IV Hamiltonian for finite-n GUE
# Route B': two-point Fredholm anchoring (no H'')
##############################

using OrdinaryDiffEq
using FastGaussQuadrature
using LinearAlgebra
using Printf

# ============================================================
# 0. ASYMPTOTIC COEFFICIENTS & TAIL (only for root selection)
# ============================================================

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

"""
    get_asymptotic_qp_unit(s, n)

Large-s asymptotic branch (A=1) for root *selection only*.
"""
function get_asymptotic_qp_unit(s::Float64, n::Int)
    @assert n == 20
    Cn = 2.0^(n-1) / (sqrt(pi) * float(factorial(big(n)-1)))

    invs = 1.0/s

    # hat_q(1/s)
    hat_q = 0.0
    pow = 1.0
    for c in Q_COEFFS_20
        hat_q += c * pow
        pow *= invs
    end

    # hat_p(1/s)
    hat_p = 0.0
    pow = 1.0
    for c in P_COEFFS_20
        hat_p += c * pow
        pow *= invs
    end

    q_val = Cn * s^(2n) * exp(-s*s) * hat_q
    p_val = hat_p
    return float(q_val), float(p_val)
end

# ============================================================
# 1. Hamiltonian system and Hermite Fredholm determinant
# ============================================================

"""
    H_hamiltonian(q, p, s, n)

Hamiltonian for finite-n GUE PIV:
H = (2p - q - 2s) p q + n q
"""
H_hamiltonian(q, p, s, n) = (2p - q - 2s) * p * q + n * q

"""
    piv_ham_system!(dY, Y, n, s)

Y = (q, p, ℓ). dℓ/ds = H.
"""
function piv_ham_system!(dY, Y, n, s)
    q  = Y[1]
    pY = Y[2]
    ℓ  = Y[3]

    dqds = q * (4.0*pY - q - 2.0*s)
    dpds = 2.0*pY * (q + s - pY) - n
    dℓds = H_hamiltonian(q, pY, s, n)

    dY[1] = dqds
    dY[2] = dpds
    dY[3] = dℓds
end

# ---------- Fredholm Nyström side (same as your code) ----------

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

function K_hermite(n::Int, s::Float64, t::Float64)
    if s == t
        _, _, sumsq = hermite_phi_chain(n, s)
        return sumsq
    end
    φnm1_s, φn_s, _ = hermite_phi_chain(n, s)
    φnm1_t, φn_t, _ = hermite_phi_chain(n, t)
    return sqrt(n/2) * (φnm1_s*φn_t - φn_s*φnm1_t) / (s - t)
end

tail_len(n::Int) = n ≤ 50 ? 8.0 : (n ≤ 500 ? 10.0 : 12.0)

function fredholm_cdf_gue_nystrom(n::Int, s::Float64,
                                  R_off::Float64,
                                  Nquad::Int)
    L = R_off
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

# ============================================================
# 2. New anchor: two-point Fredholm shooting
# ============================================================

"""
    anchor_qp_two_point(n, s0, Δ, R_off, Nquad;
                        reltol=1e-10, maxit=8)

Find (q0,p0) at s0 s.t. integrating the Hamiltonian IVP to
s1 = s0+Δ and s2 = s0-Δ reproduces the Fredholm log CDF
at those points.

Returns (q0, p0, ℓ0).
"""
function anchor_qp_two_point(n::Int,
                             s0::Float64,
                             Δ::Float64,
                             R_off::Float64,
                             Nquad::Int;
                             reltol::Float64 = 1e-10,
                             maxit::Int = 8)

    # Fredholm targets
    ℓ0 = log(fredholm_cdf_gue_nystrom(n, s0,   R_off, Nquad))
    ℓ1 = log(fredholm_cdf_gue_nystrom(n, s0+Δ, R_off, Nquad))
    ℓ2 = log(fredholm_cdf_gue_nystrom(n, s0-Δ, R_off, Nquad))

    # Initial guess from asymptotics (amplitude 1)
    q_asym, p_asym = get_asymptotic_qp_unit(s0, n)
    q0, p0 = q_asym, p_asym

    @printf("\n--- Two-point anchor at s0 = %.4f (Δ=%.3f) ---\n", s0, Δ)
    @printf("log F_n(s0) ≈ %.12e\n", ℓ0)

    for it in 1:maxit
        # Integrate forward to s1
        Y0 = [q0, p0, ℓ0]
        prob_f = ODEProblem(piv_ham_system!, Y0, (s0, s0+Δ), n)
        sol_f  = solve(prob_f, Rodas5(), reltol=1e-11, abstol=1e-13)
        ℓ1_model = sol_f(s0+Δ)[3]

        # Integrate backward to s2
        prob_b = ODEProblem(piv_ham_system!, Y0, (s0, s0-Δ), n)
        sol_b  = solve(prob_b, Rodas5(), reltol=1e-11, abstol=1e-13)
        ℓ2_model = sol_b(s0-Δ)[3]

        r1 = ℓ1_model - ℓ1
        r2 = ℓ2_model - ℓ2
        resnorm = sqrt(r1^2 + r2^2)

        @printf("  iter %d: |res| = %.3e\n", it, resnorm)

        if resnorm < reltol
            @printf("  converged: q0 ≈ %.8e, p0 ≈ %.8e\n", q0, p0)
            return q0, p0, ℓ0
        end

        # Finite-difference Jacobian in (q0,p0)
        δq = 1e-4 * max(1.0, abs(q0))
        δp = 1e-4 * max(1.0, abs(p0))

        # perturb q
        Y0q = [q0+δq, p0, ℓ0]
        sol_fq = solve(ODEProblem(piv_ham_system!, Y0q, (s0, s0+Δ), n),
                       Rodas5(), reltol=1e-11, abstol=1e-13)
        sol_bq = solve(ODEProblem(piv_ham_system!, Y0q, (s0, s0-Δ), n),
                       Rodas5(), reltol=1e-11, abstol=1e-13)
        ℓ1q = sol_fq(s0+Δ)[3]
        ℓ2q = sol_bq(s0-Δ)[3]
        J11 = (ℓ1q - ℓ1_model)/δq
        J21 = (ℓ2q - ℓ2_model)/δq

        # perturb p
        Y0p = [q0, p0+δp, ℓ0]
        sol_fp = solve(ODEProblem(piv_ham_system!, Y0p, (s0, s0+Δ), n),
                       Rodas5(), reltol=1e-11, abstol=1e-13)
        sol_bp = solve(ODEProblem(piv_ham_system!, Y0p, (s0, s0-Δ), n),
                       Rodas5(), reltol=1e-11, abstol=1e-13)
        ℓ1p = sol_fp(s0+Δ)[3]
        ℓ2p = sol_bp(s0-Δ)[3]
        J12 = (ℓ1p - ℓ1_model)/δp
        J22 = (ℓ2p - ℓ2_model)/δp

        J = [J11 J12; J21 J22]
        r = [-r1; -r2]

        # Solve J * δ = -r
        δ = J \ r
        q0 += δ[1]
        p0 += δ[2]
    end

    error("anchor_qp_two_point: did not converge in $maxit iterations")
end

# ============================================================
# 3. IVP solver using two-point anchored (q0,p0)
# ============================================================

function solve_piv_hamiltonian_ivp_twopoint(n::Int,
                                            s_target::Float64,
                                            s_anchor::Float64,
                                            Δ::Float64,
                                            R_off::Float64,
                                            Nquad::Int;
                                            n_save::Int = 80)

    # 0) Anchor (q0,p0,ℓ0) using two-point match
    q0, p0, ℓ0 = anchor_qp_two_point(n, s_anchor, Δ, R_off, Nquad)

    Y0 = [q0, p0, ℓ0]

    println()
    println("--- Solving PIV Hamiltonian IVP (two-point anchored) ---")
    @printf("n = %d, s_target = %.3f, s_anchor = %.3f\n\n",
            n, s_target, s_anchor)

    # 1) Integrate from s_anchor down to s_target
    prob = ODEProblem(piv_ham_system!, Y0, (s_anchor, s_target), n)
    sol  = solve(prob, Rodas5(), reltol=1e-11, abstol=1e-13)

    s_all = collect(LinRange(s_target, s_anchor, n_save))
    ℓ_all = similar(s_all)

    for (i, s) in pairs(s_all)
        if s < minimum(sol.t) || s > maximum(sol.t)
            error("s=$s outside solution interval")
        end
        Y = sol(s)
        ℓ_all[i] = Y[3]
    end

    return s_all, ℓ_all, true
end

# ============================================================
# 4. Driver: compare to Fredholm
# ============================================================

function main()
    n_val    = 20
    s_anchor = 6.40
    s_target = 5.00
    Δ        = 0.15
    R_off    = tail_len(n_val)
    Nquad    = 260

    s_all, ℓ_all, success =
        solve_piv_hamiltonian_ivp_twopoint(n_val, s_target, s_anchor,
                                           Δ, R_off, Nquad;
                                           n_save = 60)

    if !success
        println("WARNING: IVP solver signaled failure.")
    end

    F_piv  = exp.(ℓ_all)
    F_fred = [fredholm_cdf_gue_nystrom(n_val, s, R_off, Nquad) for s in s_all]

    diffs = abs.(F_piv .- F_fred)
    maxdiff, idx = findmax(diffs)

    println("==============================================")
    println("Hamiltonian PIV (two-point anchored) vs Fredholm")
    println("n = $n_val, s ∈ [$(minimum(s_all)), $(maximum(s_all))]")
    println("==============================================")
    @printf("%8s  %15s  %15s  %10s\n", "s", "F_PIV", "F_Fredholm", "diff")
    @printf("%s\n", "-"^60)
    for j in 1:3:length(s_all)
        @printf("%8.3f  %15.8e  %15.8e  %10.3e\n",
                s_all[j], F_piv[j], F_fred[j], diffs[j])
    end
    println()

    @printf("Max |F_PIV - F_Fred| ≈ %.3e at s ≈ %.6f\n",
            maxdiff, s_all[idx])
end

main()
