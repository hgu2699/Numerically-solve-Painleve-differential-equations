# piv_residual_diagnostic.jl
# Step 2: Check that R_n(s) = -d/ds log F_n(s) from Fredholm
# approximately satisfies the Forrester–Witte PIV σ-equation:
#
#   (R'')^2 + 4 (R')^2 (R' + 2n) - 4 (s R' - R)^2 = 0.
#
# Uses Nyström method for the finite-n GUE gap probability.

import Pkg
for p in ["FastGaussQuadrature","LinearAlgebra","Statistics","Plots"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature, LinearAlgebra, Statistics, Plots
gr()

# ---------------------------
# Orthonormal Hermite functions (weight e^{-x^2/2})
# ---------------------------
function hermite_phi_chain(n::Int, x::Float64)
    # returns (φ_{n-1}(x), φ_n(x), sum_{k=0}^{n-1} φ_k(x)^2)
    ϕ0 = pi^(-0.25) * exp(-0.5*x*x)
    if n == 1
        return (ϕ0, sqrt(2.0)*x*ϕ0, ϕ0^2)
    end
    ϕ1 = sqrt(2.0)*x*ϕ0
    sumsq = ϕ0^2
    if n == 2
        sumsq += ϕ1^2
        return (ϕ1, sqrt(1.0)*x*ϕ1 - 1.0*ϕ0, sumsq)
    end
    ϕkm1, ϕk = ϕ0, ϕ1
    sumsq += ϕ1^2
    for k in 1:(n-2) # will end at k = n-2 producing ϕ_{n-1}
        α = sqrt(2.0/(k+1))
        β = sqrt(k/(k+1))
        ϕkp1 = α*x*ϕk - β*ϕkm1
        ϕkm1, ϕk = ϕk, ϕkp1
        sumsq += ϕk^2
    end
    # now ϕk = φ_{n-1}. One more step for φ_n:
    α = sqrt(2.0/n)
    β = sqrt((n-1)/n)
    ϕn = α*x*ϕk - β*ϕkm1
    return (ϕk, ϕn, sumsq)
end

# Christoffel–Darboux Hermite kernel with diagonal fallback
# K_n(s,t) = sqrt(n/2) * [φ_{n-1}(s) φ_n(t) - φ_n(s) φ_{n-1}(t)] / (s - t),   s≠t
# K_n(s,s) = ∑_{k=0}^{n-1} φ_k(s)^2
function K_hermite(n::Int, s::Float64, t::Float64)
    if s == t
        _, _, sumsq = hermite_phi_chain(n, s)
        return sumsq
    end
    φnm1_s, φn_s, _ = hermite_phi_chain(n, s)
    φnm1_t, φn_t, _ = hermite_phi_chain(n, t)
    return sqrt(n/2) * (φnm1_s*φn_t - φn_s*φnm1_t) / (s - t)
end

# Nyström tail length
tail_len(n::Int) = n ≤ 50 ? 8.0 : (n ≤ 500 ? 10.0 : 12.0)

# ---------------------------
# log Fredholm determinant log F_n(s) via Nyström
# (returns logF directly to avoid underflow)
# ---------------------------
function logF_n_fredholm(n::Int, s::Float64; N::Int=240, L::Float64=tail_len(n))
    # quadrature nodes on [s, s+L]
    z, w = gausslegendre(N)          # (-1,1)
    u  = (z .+ 1.0) .* (L/2)         # (0,L)
    dt = (L/2) .* w
    t  = s .+ u

    sq = sqrt.(dt)
    A  = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N
        tj = t[j]
        for i in 1:N
            A[i,j] = sq[i] * K_hermite(n, t[i], tj) * sq[j]
        end
    end
    λ = eigvals(Matrix(I - A))
    λ = clamp.(real.(λ), eps(), 1.0)
    return sum(log, λ)    # log F_n(s)
end

# ---------------------------
# High-order finite-difference derivatives on uniform grid
# ---------------------------
function deriv1_4th(f::AbstractVector{<:Real}, h::Float64)
    n = length(f)
    df = zeros(Float64, n)
    # interior: 4th-order central
    @inbounds for i in 3:(n-2)
        df[i] = (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12h)
    end
    # boundaries: 2nd-order one-sided
    df[1]     = (-3f[1] + 4f[2] - f[3]) / (2h)
    df[2]     = (-3f[2] + 4f[3] - f[4]) / (2h)
    df[n]     = ( 3f[n] - 4f[n-1] + f[n-2]) / (2h)
    df[n-1]   = ( 3f[n-1] - 4f[n-2] + f[n-3]) / (2h)
    return df
end

function deriv2_4th(f::AbstractVector{<:Real}, h::Float64)
    n = length(f)
    d2 = zeros(Float64, n)
    # interior: 4th-order central second derivative
    @inbounds for i in 3:(n-2)
        d2[i] = (-f[i+2] + 16f[i+1] - 30f[i] + 16f[i-1] - f[i-2]) / (12h^2)
    end
    # boundaries: 2nd-order
    d2[1]   = (f[3] - 2f[2] + f[1]) / (h^2)
    d2[2]   = (f[4] - 2f[3] + f[2]) / (h^2)
    d2[n]   = (f[n] - 2f[n-1] + f[n-2]) / (h^2)
    d2[n-1] = (f[n-1] - 2f[n-2] + f[n-3]) / (h^2)
    return d2
end

# ---------------------------
# PIV σ-form residual diagnostic
# ---------------------------
function piv_residual_diagnostic(n::Int;
                                 window_half::Float64=3.0,
                                 NquadFD::Int=260,
                                 npts::Int=801)

    s_edge = sqrt(2.0*n)
    smin, smax = s_edge - window_half, s_edge + window_half
    println("PIV residual diagnostic: n=$n, domain = [$smin, $smax], npts=$npts")

    sgrid = collect(range(smin, smax; length=npts))
    h = sgrid[2] - sgrid[1]

    # log F_n(s) on grid
    logF = zeros(Float64, npts)
    for (k, s) in pairs(sgrid)
        logF[k] = logF_n_fredholm(n, s; N=NquadFD, L=tail_len(n))
    end

    # derivatives of log F
    logFp  = deriv1_4th(logF, h)   # (log F)'
    logFpp = deriv2_4th(logF, h)   # (log F)''

    # R(s) = - (log F)' ; R'(s) = - (log F)''
    R   = .-logFp
    Rp  = .-logFpp
    Rpp = deriv1_4th(Rp, h)        # R'' ≈ derivative of R'

    # PIV σ-form residual:
    # Res(s) = (R'')^2 + 4 (R')^2 (R' + 2n) - 4 (s R' - R)^2
    Res = similar(R)
    @inbounds for k in 1:npts
        Rppk = Rpp[k]
        Rpk  = Rp[k]
        Rk   = R[k]
        sk   = sgrid[k]
        Res[k] = Rppk^2 + 4.0*Rpk^2*(Rpk + 2.0*n) - 4.0*(sk*Rpk - Rk)^2
    end

    # Ignore a small boundary strip where high-order stencils are less accurate
    k_lo = 5
    k_hi = npts - 4
    s_int   = sgrid[k_lo:k_hi]
    Res_int = Res[k_lo:k_hi]

    max_res  = maximum(abs.(Res_int))
    mean_res = mean(abs.(Res_int))

    println("Max |Res(s)| on interior grid = $max_res")
    println("Mean |Res(s)| on interior grid = $mean_res")

    # Plot |Res(s)| on the interior
    plt = plot(s_int, abs.(Res_int), lw=2,
               xlabel="s", ylabel="|Res(s)|",
               title="PIV σ-form residual for finite-n GUE (n=$n)")
    display(plt)
    savefig(plt, "piv_residual_n$(n).png")

    return (sgrid=sgrid, Res=Res, max_res=max_res, mean_res=mean_res, plot=plt)
end

# ---------------------------
# Example usage
# ---------------------------
function main()
    piv_residual_diagnostic(10; window_half=3.5, NquadFD=240, npts=801)
    piv_residual_diagnostic(20; window_half=3.2, NquadFD=260, npts=801)
end

main()
