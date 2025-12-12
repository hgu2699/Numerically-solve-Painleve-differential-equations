# finite_n_pvi_jue_autoanchors.jl
# Finite-N JUE: Fredholm (Gram/Nyström) vs Painlevé VI
# with automatically chosen anchors in y = -log(1-t).

import Pkg
for p in ["FastGaussQuadrature","LinearAlgebra","Statistics","Plots","SpecialFunctions","Printf"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature, LinearAlgebra, Statistics, Plots, SpecialFunctions
using Printf: @sprintf
gr()

# ------------------------------------------------------------
# Global numerical guards / design constants
# ------------------------------------------------------------

const S_GUARD     = 1e-10       # keep s strictly inside (-1,1)
const ANCHOR_EPS  = 1e-12       # tiny slack for monotonic guards
const P_MIN       = 0.10        # left quantile for window (10%)
const P_MAX       = 1 - 1e-8    # desired right quantile for window
const ΔY_ANCHOR   = 0.15        # target spacing of anchors in y
const GRID_FACTOR = 4           # y-grid spacing = ΔY_ANCHOR / GRID_FACTOR

# ------------------------------------------------------------
# Jacobi orthonormal system and Gram/Nyström Fredholm
# ------------------------------------------------------------

@inline function log_h_jacobi(n::Int, a::Float64, b::Float64)
    (a+b+1.0)*log(2.0) +
    loggamma(n + a + 1.0) + loggamma(n + b + 1.0) -
    log(2.0*n + a + b + 1.0) -
    loggamma(n + 1.0) - loggamma(n + a + b + 1.0)
end

function inv_sqrt_h_vec(N::Int, a::Float64, b::Float64)
    v = Vector{Float64}(undef, N)
    @inbounds for n in 0:N-1
        v[n+1] = exp(-0.5*log_h_jacobi(n, a, b))
    end
    return v
end

# Orthonormal Jacobi row φ_k(x), k=0..N-1
function jacobi_phi_row!(row::AbstractVector{Float64}, x::Float64,
                         N::Int, a::Float64, b::Float64,
                         inv_sqrt_h::Vector{Float64})
    @assert length(row) == N
    if !(x > -1.0 && x < 1.0)
        error("Jacobi support is (-1,1). Got x=$x")
    end
    sqrtw = (1 - x)^(0.5*a) * (1 + x)^(0.5*b)

    # P_0
    Pkm1 = 1.0
    row[1] = Pkm1 * inv_sqrt_h[1] * sqrtw
    if N == 1; return; end

    # P_1
    Pk = 0.5*((a - b) + (a + b + 2.0)*x)
    row[2] = Pk * inv_sqrt_h[2] * sqrtw
    if N == 2; return; end

    # 3-term recurrence (DLMF 18.9.5)
    for n in 1:(N-2)  # produces P_{n+1}
        c1 = 2.0*(n+1.0)*(n + a + b + 1.0)*(2.0*n + a + b)
        c2 = (2.0*n + a + b + 1.0)*((2.0*n + a + b + 2.0)*(2.0*n + a + b)*x + a^2 - b^2)
        c3 = 2.0*(n + a)*(n + b)*(2.0*n + a + b + 2.0)
        Pkp1 = (c2*Pk - c3*Pkm1)/c1
        Pkm1, Pk = Pk, Pkp1
        row[n+2] = Pk * inv_sqrt_h[n+2] * sqrtw
    end
end

# Build Φ matrix: Φ[i,k] = φ_k(t_i), k=0..N-1
function Phi_jacobi(t::Vector{Float64}, N::Int, a::Float64, b::Float64)
    invsh = inv_sqrt_h_vec(N, a, b)
    Φ = Matrix{Float64}(undef, length(t), N)
    @inbounds for i in eachindex(t)
        jacobi_phi_row!(view(Φ, i, :), t[i], N, a, b, invsh)
    end
    return Φ
end

# Fredholm determinant on (s,1) via Gram/Nyström
function F_N_fredholm_JUE(N::Int, a::Float64, b::Float64, s::Float64; Nquad::Int=260)
    s_eff = min(max(s, -1.0 + S_GUARD), 1.0 - S_GUARD)
    if !(s_eff > -1.0 && s_eff < 1.0)
        error("s must lie in (-1,1). Got s=$(s)")
    end
    z, w = gausslegendre(Nquad)
    t  = 0.5 .* ((1.0 - s_eff).*z .+ (1.0 + s_eff))  # map (-1,1) -> (s_eff,1)
    dt = 0.5 .* (1.0 - s_eff) .* w
    Φ  = Phi_jacobi(t, N, a, b)
    WΦ = Φ .* sqrt.(dt)               # row-scale by sqrt(dt)
    A  = WΦ * transpose(WΦ)           # Nyström matrix (rank N kernel)
    λ  = eigvals(Matrix(I - A))
    λ  = clamp.(real.(λ), eps(), 1.0) # numeric guard
    return exp(sum(log, λ))
end

F_N_vec_JUE(N::Int, a::Float64, b::Float64, svals::AbstractVector{<:Real}; Nquad::Int=260) =
    [F_N_fredholm_JUE(N, a, b, Float64(s); Nquad=Nquad) for s in svals]

# ------------------------------------------------------------
# Local LS fit: log F(s) near s0 → σ(s0), σ'(s0)
# ------------------------------------------------------------

function sigma_fit_logF_at(s0::Float64, spts::Vector{Float64}, Fvals::Vector{Float64})
    u = spts .- s0
    y = log.(Fvals)
    M = hcat(u.^0, u, u.^2, u.^3, u.^4)    # degree-4 local fit

    du   = sort(abs.(diff(sort(spts))))
    hmed = isempty(du) ? 1e-3 : max(median(du), 1e-6)
    σw   = 2.0*hmed
    w    = exp.(-(u./σw).^2)
    W    = Diagonal(w)

    c  = (M' * W * M) \ (M' * W * y)
    σ  = c[2]         # d/ds log F at s0
    σp = 2c[3]        # d^2/ds^2 log F at s0
    return (; σ, σp)
end

# Boundary-safe stencil around s0
function safe_stencil_points(s0::Float64, half_step::Float64;
                             lo::Float64=-1.0+S_GUARD, hi::Float64=1.0-S_GUARD)
    base = (-3:3) .* half_step
    scale = 1.0
    while true
        δ = base .* scale
        spts = s0 .+ δ
        spts = spts[(spts .> lo) .& (spts .< hi)]
        if length(spts) >= 5
            return spts
        end
        scale *= 0.7
        if scale < 1e-6
            spts = clamp.(s0 .+ base, lo+S_GUARD, hi-S_GUARD)
            return unique(sort(spts))
        end
    end
end

# ------------------------------------------------------------
# Painlevé VI σ-form → equation for f''(t),
# with f(t) = F_N(s), t = (s+1)/2.
# ------------------------------------------------------------

@inline function pvi_rhs_sq(t::Float64, f::Float64, fp::Float64,
                            v1::Float64, v2::Float64, v4::Float64)
    term1 = 4.0*t*(1.0-t)*(fp^3)
    term2 = 4.0*(1.0 - 2.0*t)*(fp^2)*f
    term3 = 4.0*fp*(f^2)
    term4 = -4.0*(f^2)*(v1^2)
    term5 = (fp^2) * ( 4.0*t*(v1^2)/(1.0 - t) - ( (v2 - v4)^2 )/4.0 - 4.0*t*v1*v4 )
    term6 = 4.0*f*fp*( -v1^2 + 2.0*t*v1^2 + v1*v4 )
    return term1 - term2 - term3 + term4 - term5 - term6
end

@inline function choose_fpp_pvi(prev_fpp::Float64, t::Float64, f::Float64, fp::Float64,
                                v1::Float64, v2::Float64, v4::Float64)
    R = pvi_rhs_sq(t, f, fp, v1, v2, v4)
    R ≤ 0 && return 0.0
    denom = t*(1.0 - t)
    denom = abs(denom) > eps() ? denom : (denom ≥ 0 ? eps() : -eps())
    r = sqrt(R) / denom
    c1 =  sign(prev_fpp) * r
    c2 = -sign(prev_fpp) * r
    return abs(c1 - prev_fpp) ≤ abs(c2 - prev_fpp) ? c1 : c2
end

# One step with stability + monotonicity guards
function step_f_pvi_guarded!(f::Float64, fp::Float64,
                             t_now::Float64, t_next::Float64,
                             v1::Float64, v2::Float64, v4::Float64,
                             prev_fpp::Float64;
                             max_halves::Int=10,
                             cap_coeff::Float64=0.12)

    tL, tR = t_now, t_next
    fL, fpL, fppL = f, fp, prev_fpp

    # cap step near t→1 to avoid jumping through stiff layer
    while abs(tR - tL) > cap_coeff*(1.0 - tL) && max_halves > 0
        t_mid = tL + 0.5*(tR - tL)
        fL, fpL, fppL = step_f_pvi_guarded!(fL, fpL, tL, t_mid,
                                            v1, v2, v4, fppL;
                                            max_halves=max_halves-1,
                                            cap_coeff=cap_coeff)
        tL = t_mid
    end

    h = tR - tL
    fpp = choose_fpp_pvi(fppL, tL, fL, fpL, v1, v2, v4)
    f_new  = fL  + h*fpL + 0.5*h^2*fpp
    fp_mid = fpL + 0.5*h*fpp
    fpp2   = choose_fpp_pvi(fpp, 0.5*(tL+tR), f_new, fp_mid, v1, v2, v4)
    fp_new = fpL + h*fpp2

    # Direction-aware monotonicity guard
    if h > 0
        # moving right (toward t=1): F must be nondecreasing
        if f_new < fL
            if max_halves > 0
                t_mid = tL + 0.5*h
                return step_f_pvi_guarded!(fL, fpL, tL, t_mid,
                                           v1, v2, v4, fppL;
                                           max_halves=max_halves-1,
                                           cap_coeff=cap_coeff)
            else
                f_new  = fL
                fp_new = max(fp_new, 0.0)
            end
        end
    elseif h < 0
        # moving left (away from t=1): F must be nonincreasing
        if f_new > fL
            if max_halves > 0
                t_mid = tL + 0.5*h
                return step_f_pvi_guarded!(fL, fpL, tL, t_mid,
                                           v1, v2, v4, fppL;
                                           max_halves=max_halves-1,
                                           cap_coeff=cap_coeff)
            else
                f_new  = fL
                fp_new = min(fp_new, 0.0)
            end
        end
    end

    f_new = clamp(f_new, 0.0, 1.0)
    return f_new, fp_new, fpp2
end

# ------------------------------------------------------------
# Quantile-based s-window: solve F_N(s) = p by bisection
# with robust clipping of p into [F_min, F_max]
# ------------------------------------------------------------

function find_s_for_CDF_quantile(N::Int, a::Float64, b::Float64, p::Float64;
                                 Nquad::Int=260)
    s_lo = -0.999999
    s_hi =  0.999999

    Flo = F_N_fredholm_JUE(N, a, b, s_lo; Nquad=Nquad)
    Fhi = F_N_fredholm_JUE(N, a, b, s_hi; Nquad=Nquad)

    # Clip p into [Flo, Fhi] with a small safety margin
    if p > Fhi
        p_clip = max(Fhi - 1e-10, Flo + 1e-10)
        @warn "Requested quantile p=$p but numerical CDF max is Fhi=$Fhi; clipping to p=$p_clip"
        p = p_clip
    elseif p < Flo
        p_clip = min(Flo + 1e-10, Fhi - 1e-10)
        @warn "Requested quantile p=$p but numerical CDF min is Flo=$Flo; clipping to p=$p_clip"
        p = p_clip
    end

    # Standard bisection
    for _ in 1:60
        s_mid = 0.5*(s_lo + s_hi)
        Fmid  = F_N_fredholm_JUE(N, a, b, s_mid; Nquad=Nquad)
        if Fmid < p
            s_lo, Flo = s_mid, Fmid
        else
            s_hi, Fhi = s_mid, Fmid
        end
    end
    return s_hi
end

# ------------------------------------------------------------
# Build automatic anchors in y = -log(1-t)
# ------------------------------------------------------------

function build_auto_anchors(N::Int, a::Float64, b::Float64;
                            p_min::Float64=P_MIN,
                            p_max::Float64=P_MAX,
                            Δy_anchor::Float64=ΔY_ANCHOR,
                            Nquad_quant::Int=260,
                            Nquad_anchor::Int=300)

    # Quantile-based window in s (robust p is handled inside finder)
    s_lo = find_s_for_CDF_quantile(N, a, b, p_min; Nquad=Nquad_quant)
    s_hi = find_s_for_CDF_quantile(N, a, b, p_max; Nquad=Nquad_quant)
    s_hi = min(s_hi, 1.0 - 1e-6)

    # Convert to t, y
    t_min = max(0.5*(s_lo + 1.0), 0.0 + 1e-12)
    t_max = min(0.5*(s_hi + 1.0), 1.0 - 1e-12)
    y_min = -log(1.0 - t_min)
    y_max = -log(1.0 - t_max)

    nanchors = max(5, Int(ceil((y_max - y_min)/Δy_anchor)) + 1)
    y_anc    = range(y_min, y_max; length=nanchors)
    t_anc    = 1 .- exp.(-y_anc)
    s_anc    = 2 .* t_anc .- 1

    lo = -1.0 + S_GUARD
    hi =  1.0 - S_GUARD

    anchors = Vector{NamedTuple}(undef, nanchors)
    for j in 1:nanchors
        s0 = clamp(s_anc[j], lo+1e-12, hi-1e-12)

        # local half-step in s based on neighbours
        if j == 1
            half_step = 0.4*abs(s_anc[2] - s_anc[1])
        elseif j == nanchors
            half_step = 0.4*abs(s_anc[end] - s_anc[end-1])
        else
            half_step = 0.2*abs(s_anc[j+1] - s_anc[j-1])
        end

        spts = safe_stencil_points(s0, half_step; lo=lo, hi=hi)
        Fpts = F_N_vec_JUE(N, a, b, spts; Nquad=Nquad_anchor)
        fit  = sigma_fit_logF_at(s0, spts, Fpts)
        F0   = clamp(F_N_fredholm_JUE(N, a, b, s0; Nquad=Nquad_anchor+40), 0.0, 1.0)

        t0 = 0.5*(s0 + 1.0)
        y0 = -log(1.0 - t0)

        anchors[j] = (s0=s0, t0=t0, y0=y0, F0=F0, σ=fit.σ, σp=fit.σp)
    end

    return anchors
end

# ------------------------------------------------------------
# Main solver: PVI with automatic anchors
# ------------------------------------------------------------

function jue_pvi_locked_autoanchors(; N::Int=20, a::Float64=0.0, b::Float64=0.0,
                                     NquadFD::Int=320)

    @info "Building automatic anchors for JUE (N=$N, a=$a, b=$b)…"
    anchors = build_auto_anchors(N, a, b;
                                 Nquad_quant=max(240, NquadFD-40),
                                 Nquad_anchor=NquadFD)

    nanchors = length(anchors)
    s_min = anchors[1].s0
    s_max = anchors[end].s0
    y_min = anchors[1].y0
    y_max = anchors[end].y0

    Δy_grid = ΔY_ANCHOR / GRID_FACTOR
    ny  = max(400, Int(ceil((y_max - y_min)/Δy_grid)) + 1)
    ygrid = range(y_min, y_max; length=ny)
    tgrid = 1 .- exp.(-ygrid)
    sgrid = 2 .* tgrid .- 1

    # Map anchors to nearest grid indices
    anch_grid_idx = Vector{Int}(undef, nanchors)
    F0_vec = Vector{Float64}(undef, nanchors)
    for j in 1:nanchors
        _, idx = findmin(abs.(ygrid .- anchors[j].y0))
        anch_grid_idx[j] = idx
        F0_vec[j] = anchors[j].F0
    end

    # Base anchor: F ≈ 0.5
    j0 = argmin(abs.(F0_vec .- 0.5))
    base = anchors[j0]
    k0   = anch_grid_idx[j0]

    @info "Base anchor j0=$j0 at s≈$(round(base.s0,digits=6)), F0≈$(round(base.F0,digits=4))"

    # PVI parameters
    v1 = N + 0.5*(a + b)
    v2 = 0.5*(a + b)
    v4 = 0.5*(b - a)

    ng = length(sgrid)
    fvals  = fill(NaN, ng)
    fpvals = fill(NaN, ng)

    # Seed at base anchor
    fvals[k0]  = base.F0
    fpvals[k0] = 2.0 * base.σ * base.F0
    fpp0       = 4.0 * (base.σp + base.σ^2) * base.F0

    # ---------- Left pass: from base to smallest y ----------
    fpp = fpp0
    for j in j0:-1:2
        k_start = anch_grid_idx[j]
        k_end   = anch_grid_idx[j-1]

        if isnan(fvals[k_start])
            fvals[k_start]  = anchors[j].F0
            fpvals[k_start] = 2.0 * anchors[j].σ * anchors[j].F0
            fpp = 4.0 * (anchors[j].σp + anchors[j].σ^2) * anchors[j].F0
        end

        for k in k_start:-1:(k_end+1)
            fvals[k-1], fpvals[k-1], fpp =
                step_f_pvi_guarded!(fvals[k], fpvals[k], tgrid[k], tgrid[k-1],
                                    v1, v2, v4, fpp; cap_coeff=0.10)
        end

        anch = anchors[j-1]
        idx  = anch_grid_idx[j-1]
        fvals[idx]  = anch.F0
        fpvals[idx] = 2.0 * anch.σ * anch.F0
        fpp         = 4.0 * (anch.σp + anch.σ^2) * anch.F0
    end

    # ---------- Right pass: from base to largest y ----------
    fpp = fpp0
    for j in j0:(nanchors-1)
        k_start = anch_grid_idx[j]
        k_end   = anch_grid_idx[j+1]

        if isnan(fvals[k_start])
            fvals[k_start]  = anchors[j].F0
            fpvals[k_start] = 2.0 * anchors[j].σ * anchors[j].F0
            fpp = 4.0 * (anchors[j].σp + anchors[j].σ^2) * anchors[j].F0
        end

        for k in k_start:(k_end-1)
            fvals[k+1], fpvals[k+1], fpp =
                step_f_pvi_guarded!(fvals[k], fpvals[k], tgrid[k], tgrid[k+1],
                                    v1, v2, v4, fpp; cap_coeff=0.10)
        end

        anch = anchors[j+1]
        idx  = anch_grid_idx[j+1]
        fvals[idx]  = anch.F0
        fpvals[idx] = 2.0 * anch.σ * anch.F0
        fpp         = 4.0 * (anch.σp + anch.σ^2) * anch.F0
    end

    F_pvi = fvals
    F_fd  = F_N_vec_JUE(N, a, b, sgrid; Nquad=NquadFD)

    # Plots
    plt1 = plot(sgrid, F_fd, lw=2, label="Fredholm (Jacobi, Gram)",
                xlabel="s", ylabel="CDF",
                title="JUE largest eigenvalue (auto anchors) N=$(N), a=$(a), b=$(b)")
    plot!(plt1, sgrid, F_pvi, lw=2, ls=:dash, label="Painlevé VI (auto-locked)")
    savefig(plt1, "jue_autoanchors_F_N$(N)_a$(round(a,digits=2))_b$(round(b,digits=2)).png")
    display(plt1)

    plt2 = plot(sgrid, abs.(F_fd .- F_pvi), lw=2, label="|Δ|",
                xlabel="s", ylabel="abs error",
                title="|Fᶠʳᵉᵈ − Fᴾⱽᴵ| (auto anchors) N=$(N), a=$(a), b=$(b)")
    savefig(plt2, "jue_autoanchors_err_N$(N)_a$(round(a,digits=2))_b$(round(b,digits=2)).png")
    display(plt2)

    maxerr = maximum(abs.(F_fd .- F_pvi))
    @info "AUTO ANCHORS  N=$N, a=$a, b=$b: max |Δ| = $maxerr"

    return (sgrid=sgrid, F_fd=F_fd, F_pvi=F_pvi,
            anchors=anchors, ygrid=ygrid,
            plt1=plt1, plt2=plt2, maxerr=maxerr)
end

# ------------------------------------------------------------
# Suite for N = 20,50,100,300 and (a,b) = (0,0),(2,0),(0,3),(2,3)
# ------------------------------------------------------------

function run_jue_suite_autoanchors()
    Ns  = [20, 50, 100, 300]
    ABs = [(0.0,0.0), (2.0,0.0), (0.0,3.0), (2.0,3.0)]

    for N in Ns, (a,b) in ABs
        NquadFD = max(260, 260 + Int(round(0.6N)))  # mild N-dependent quadrature
        @info "============================================================"
        @info "Running JUE auto-anchors: N=$N, a=$a, b=$b, NquadFD=$NquadFD"
        jue_pvi_locked_autoanchors(N=N, a=a, b=b, NquadFD=NquadFD)
    end
    @info "Finished auto-anchor JUE suite. Figures saved in $(pwd())."
end

# Uncomment to run the full suite immediately:
run_jue_suite_autoanchors()
