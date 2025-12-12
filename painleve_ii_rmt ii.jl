# painleve_tw_compare.jl
# Compare Tracy–Widom GUE CDF F₂(x) via:
# (1) Hastings–McLeod Painlevé II solution
# (2) Fredholm determinant of the Airy kernel (Nyström)
#
# NOTE: the backward Painlevé II IVP from T0 is numerically delicate.
# In standard double precision this will NOT reliably give the true
# Hastings–McLeod solution, so F₂ from PII can be very wrong.
# The Fredholm determinant side is trustworthy.

import Pkg
for p in ["OrdinaryDiffEq", "FastGaussQuadrature", "SpecialFunctions",
          "QuadGK", "LinearAlgebra", "Plots"]
    if Base.find_package(p) === nothing
        try
            Pkg.add(p)
        catch
            Pkg.Registry.update()
            Pkg.add(p)
        end
    end
end

using OrdinaryDiffEq
using FastGaussQuadrature
using SpecialFunctions   # airyai, airyaiprime
using QuadGK
using LinearAlgebra
using Plots

# ---- Correct function aliases (no (t)!) ----
const Ai  = airyai
const Aip = airyaiprime

# ---------------------------
# 1) Painlevé II: q'' = x q + 2 q^3, Hastings–McLeod at +∞
# ---------------------------
function solve_pii_on_grid(xmin::Float64, xmax::Float64;
                           T0::Float64 = 12.0, N::Int = 1401)
    @assert xmin < xmax
    function pii!(du, u, p, t)  # (du,u,p,t)
        q  = u[1]
        qp = u[2]
        du[1] = qp
        du[2] = t*q + 2q^3
        return nothing
    end

    # Initial conditions at large t = T0 from Airy
    q0  = Ai(T0)
    qp0 = Aip(T0)
    u0  = [q0, qp0]

    # Integrate backward t ∈ [xmin, T0]
    prob = ODEProblem(pii!, u0, (T0, xmin))
    sol  = solve(prob, Vern7(); reltol=1e-10, abstol=1e-12,
                 saveat=range(T0, xmin; length=N))

    # Ascending grid on [xmin, xmax]
    tfull = Array(sol.t)
    sel   = findall(x -> xmin - 1e-14 ≤ x ≤ xmax + 1e-14, tfull)
    xgrid = sort(tfull[sel])

    # Interpolate with the *function* form sol(x; idxs=1)
    qvals = [sol(x; idxs=1) for x in xgrid]
    return xgrid, qvals, sol
end

# F₂ from q via ln F₂(x) = -∫_x^∞ (t-x) q(t)^2 dt
function F2_from_q(xgrid::Vector{Float64}, q::Vector{Float64};
                   T0::Float64 = 12.0)
    n  = length(xgrid)
    q2 = q .^ 2

    # cumulative trapezoids from right:
    I0 = zeros(n)   # ≈ ∫_x^{xmax} q^2
    I1 = zeros(n)   # ≈ ∫_x^{xmax} t q^2
    for k in (n-1):-1:1
        dx        = xgrid[k+1] - xgrid[k]
        f0a, f0b  = q2[k], q2[k+1]
        I0[k]     = I0[k+1] + 0.5*(f0a + f0b)*dx

        f1a, f1b  = xgrid[k]*q2[k], xgrid[k+1]*q2[k+1]
        I1[k]     = I1[k+1] + 0.5*(f1a + f1b)*dx
    end

    # Tiny Airy tail beyond T0 (super-exponentially small, but include it)
    tail_I0, _ = quadgk(t -> Ai(t)^2, T0, T0+20; rtol=1e-11, atol=1e-13)
    tail_I1, _ = quadgk(t -> t*Ai(t)^2, T0, T0+20; rtol=1e-11, atol=1e-13)

    logF = similar(xgrid)
    for (k, x) in enumerate(xgrid)
        logF[k] = - (I1[k] - x*I0[k]) - (tail_I1 - x*tail_I0)
    end
    return exp.(logF)
end

# ---------------------------
# 2) Fredholm determinant of Airy kernel via Nyström
# K(s,t) = (Ai(s)Ai'(t) - Ai'(s)Ai(t))/(s - t),
# with diagonal K(s,s) = Ai'(s)^2 - s Ai(s)^2
# Map z∈(0,1) → t = x + z/(1-z), dt = dz/(1-z)^2
# ---------------------------
function K_Airy(s::Float64, t::Float64)
    if s == t
        return Aip(s)^2 - s*Ai(s)^2
    else
        return (Ai(s)*Aip(t) - Aip(s)*Ai(t)) / (s - t)
    end
end

function F2_fredholm(x::Float64; N::Int = 80)
    z, w = gausslegendre(N)          # nodes, weights on (-1,1)
    z  = (z .+ 1.0) ./ 2.0           # → (0,1)
    w  = w ./ 2.0

    t  = x .+ z ./ (1 .- z)
    dt = w ./ (1 .- z).^2

    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N
        tj = t[j]
        for i in 1:N
            A[i,j] = sq[i] * K_Airy(t[i], tj) * sq[j]
        end
    end

    B = Matrix(I - A)
    λ = eigvals(B)
    λ = clamp.(real.(λ), eps(), 1.0)
    return exp(sum(log, λ))
end

F2_fredholm_vec(xv::AbstractVector{<:Real}; N::Int = 80) =
    [F2_fredholm(Float64(x); N=N) for x in xv]

# simple linear interpolation (for plotting on the PII grid)
function interp1(x::Vector{Float64}, y::Vector{Float64},
                 xi::Vector{Float64})
    yi = similar(xi)
    j  = 1
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
    yi
end

# ---------------------------
# 3) Run: solve, compare, and plot
# ---------------------------
xmin, xmax = -8.0, 4.0
xgrid, qvals, _ = solve_pii_on_grid(xmin, xmax; T0=12.0, N=1401)
F2_pii = F2_from_q(xgrid, qvals; T0=12.0)

x_fred = range(xmin, xmax; length=121)
@info "Computing Fredholm determinants on $(length(x_fred)) points (N=80 Nyström)…"
F2_fred = F2_fredholm_vec(collect(x_fred); N=80)
F2_fred_on_grid = interp1(collect(x_fred), F2_fred, xgrid)

plt1 = plot(xgrid, F2_pii, lw=2, label="Painlevé II → F₂(x)",
            xlabel="x", ylabel="CDF",
            title="Tracy–Widom GUE CDF: Painlevé II vs Fredholm determinant")
plot!(plt1, xgrid, F2_fred_on_grid, lw=2, ls=:dash,
      label="Fredholm Airy kernel")

plt2 = plot(xgrid, abs.(F2_pii .- F2_fred_on_grid), lw=2, label="|Δ|",
            xlabel="x", ylabel="absolute error",
            title="Absolute difference")

png1 = "tw_cdf_compare.png"
png2 = "tw_cdf_absdiff.png"
savefig(plt1, png1)
savefig(plt2, png2)
display(plt1); display(plt2)

maxerr = maximum(abs.(F2_pii .- F2_fred_on_grid))
@info "Max |difference| on grid = $(maxerr)"
@info "Saved plots: $png1 and $png2"
