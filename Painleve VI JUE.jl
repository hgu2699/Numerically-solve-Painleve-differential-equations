# finite_n_pvi_jue_locked_suite_strong2.jl
# Finite-N JUE: Fredholm (Gram/Nyström) vs Painlevé VI with robust anchor-locking.

import Pkg
for p in ["FastGaussQuadrature","LinearAlgebra","Statistics","Plots","SpecialFunctions","Printf"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature, LinearAlgebra, Statistics, Plots, SpecialFunctions
using Printf: @sprintf
gr()

const S_GUARD   = 1e-10       # keep s strictly inside (-1,1)
const ANCHOR_EPS = 1e-12      # tiny slack for anchor monotonic guard

# ---------------- Chebyshev grids (cluster at endpoints) ----------------
cheb_nodes_desc(a::Float64, b::Float64, m::Int) = begin
    xs = [cos((2k-1)*pi/(2m)) for k in 1:m]   # (-1,1), clustered to ±1
    s  = 0.5*((b-a).*xs .+ (b+a))             # map → [a,b]
    sort(s; rev=true)                          # descending
end

# ---------- Jacobi norms (log form for stability) ----------
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

# ---------- Orthonormal Jacobi row φ_k(x), k=0..N-1 ----------
function jacobi_phi_row!(row::AbstractVector{Float64}, x::Float64, N::Int, a::Float64, b::Float64, inv_sqrt_h::Vector{Float64})
    @assert length(row) == N
    if !(x > -1.0 && x < 1.0)
        error("Jacobi support is (-1,1). Got x=$x")
    end
    sqrtw = (1 - x)^(0.5*a) * (1 + x)^(0.5*b)

    # P0, P1
    Pkm1 = 1.0
    row[1] = Pkm1 * inv_sqrt_h[1] * sqrtw
    if N == 1; return; end
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

# ---------- Fredholm determinant on (s,1) via Gram/Nyström ----------
function F_N_fredholm_JUE(N::Int, a::Float64, b::Float64, s::Float64; Nquad::Int=240)
    s_eff = min(max(s, -1.0 + S_GUARD), 1.0 - S_GUARD)
    if !(s_eff > -1.0 && s_eff < 1.0)
        error("s must lie in (-1,1). Got s=$(s)")
    end
    z, w = gausslegendre(Nquad)
    # map (-1,1) -> (s_eff, 1)
    t  = 0.5 .* ((1.0 - s_eff).*z .+ (1.0 + s_eff))
    dt = 0.5 .* (1.0 - s_eff) .* w
    Φ  = Phi_jacobi(t, N, a, b)
    WΦ = Φ .* sqrt.(dt)               # row-scale by sqrt(dt)
    A  = WΦ * transpose(WΦ)           # Nyström matrix (rank N kernel)
    λ = eigvals(Matrix(I - A))
    λ = clamp.(real.(λ), eps(), 1.0)  # numeric guard
    return exp(sum(log, λ))
end

F_N_vec_JUE(N::Int, a::Float64, b::Float64, svals::AbstractVector{<:Real}; Nquad::Int=240) =
    [F_N_fredholm_JUE(N, a, b, Float64(s); Nquad=Nquad) for s in svals]

# ---------- Weighted local LS fit (log F -> σ, σ') (NO F0 here) ----------
function sigma_fit_logF(spts::Vector{Float64}, Fvals::Vector{Float64})
    s0 = spts[cld(length(spts),2)]
    u  = spts .- s0
    y  = log.(Fvals)
    M  = hcat(u.^0, u, u.^2, u.^3, u.^4)    # degree-4 local fit

    # center-weighted LS (Gaussian weights)
    du   = sort(abs.(diff(sort(spts))))
    hmed = isempty(du) ? 1e-3 : max(median(du), 1e-6)
    σw   = 2.0*hmed
    w    = exp.(-(u./σw).^2)
    W    = Diagonal(w)

    c  = (M' * W * M) \ (M' * W * y)
    σ   = c[2]         # d/ds log F at s0
    σp  = 2c[3]        # d^2/ds^2 log F at s0
    return (; s0, σ, σp)
end

# ---------- PVI in f(t)=F(t), t=(s+1)/2 ----------
# (t(1-t) f'')^2 - 4 t(1-t) (f')^3 + 4 (1-2t) (f')^2 f + 4 f' f^2 - 4 f^2 v1^2
# + (f')^2*( 4 t v1^2/(1-t) - (v2 - v4)^2/4 - 4 t v1 v4 )
# + 4 f f'*( -v1^2 + 2 t v1^2 + v1 v4 ) = 0.
@inline function pvi_rhs_sq(t::Float64, f::Float64, fp::Float64, v1::Float64, v2::Float64, v4::Float64)
    term1 = 4.0*t*(1.0-t)*(fp^3)
    term2 = 4.0*(1.0 - 2.0*t)*(fp^2)*f
    term3 = 4.0*fp*(f^2)
    term4 = -4.0*(f^2)*(v1^2)
    term5 = (fp^2) * ( 4.0*t*(v1^2)/(1.0 - t) - ( (v2 - v4)^2 )/4.0 - 4.0*t*v1*v4 )
    term6 = 4.0*f*fp*( -v1^2 + 2.0*t*v1^2 + v1*v4 )
    return term1 - term2 - term3 + term4 - term5 - term6
end

@inline function choose_fpp_pvi(prev_fpp::Float64, t::Float64, f::Float64, fp::Float64, v1::Float64, v2::Float64, v4::Float64)
    R = pvi_rhs_sq(t, f, fp, v1, v2, v4)
    R ≤ 0 && return 0.0
    denom = t*(1.0 - t)
    denom = abs(denom) > eps() ? denom : (denom ≥ 0 ? eps() : -eps())
    r = sqrt(R) / denom
    c1 =  sign(prev_fpp) * r
    c2 = -sign(prev_fpp) * r
    return abs(c1 - prev_fpp) ≤ abs(c2 - prev_fpp) ? c1 : c2
end

# --- single step with stability guards (cap, direction-aware monotonicity) ---
function step_f_pvi_guarded!(f::Float64, fp::Float64, t_now::Float64, t_next::Float64,
                             v1::Float64, v2::Float64, v4::Float64, prev_fpp::Float64;
                             max_halves::Int=10, cap_coeff::Float64=0.12)

    tL, tR = t_now, t_next
    fL, fpL, fppL = f, fp, prev_fpp

    # cap step near t→1 to avoid jumping through the stiff layer
    while abs(tR - tL) > cap_coeff*(1.0 - tL) && max_halves > 0
        t_mid = tL + 0.5*(tR - tL)
        fL, fpL, fppL = step_f_pvi_guarded!(fL, fpL, tL, t_mid, v1, v2, v4, fppL; max_halves=max_halves-1, cap_coeff=cap_coeff)
        tL = t_mid
    end

    h = tR - tL
    fpp = choose_fpp_pvi(fppL, tL, fL, fpL, v1, v2, v4)
    f_new  = fL  + h*fpL + 0.5*h^2*fpp
    fp_mid = fpL + 0.5*h*fpp
    fpp2   = choose_fpp_pvi(fpp, 0.5*(tL+tR), f_new, fp_mid, v1, v2, v4)
    fp_new = fpL + h*fpp2

    # ---- Direction-aware monotonicity guard ----
    if h > 0
        # moving right (toward t=1): F must be nondecreasing
        if f_new < fL
            if max_halves > 0
                t_mid = tL + 0.5*h
                return step_f_pvi_guarded!(fL, fpL, tL, t_mid, v1, v2, v4, fppL; max_halves=max_halves-1, cap_coeff=cap_coeff)
            else
                f_new = fL
                fp_new = max(fp_new, 0.0)
            end
        end
    elseif h < 0
        # moving left (away from t=1): F must be nonincreasing
        if f_new > fL
            if max_halves > 0
                t_mid = tL + 0.5*h
                return step_f_pvi_guarded!(fL, fpL, tL, t_mid, v1, v2, v4, fppL; max_halves=max_halves-1, cap_coeff=cap_coeff)
            else
                f_new = fL
                fp_new = min(fp_new, 0.0)
            end
        end
    end

    f_new = clamp(f_new, 0.0, 1.0)
    return f_new, fp_new, fpp2
end

# ---------- Boundary-safe 7-point stencil near (-1,1) ----------
function safe_stencil_points(s0::Float64, half_step::Float64; lo::Float64=-1.0+S_GUARD, hi::Float64=1.0-S_GUARD)
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

# ---------------- One full run (STRONG v2) ----------------
function jue_pvi_locked_strong2(; N::Int=20, a::Float64=0.0, b::Float64=0.0,
                                 NquadFD::Int=300,
                                 npts::Int=1501, nanchors::Int=141,
                                 s_margin::Float64=5e-6, window_half::Float64=0.25)

    # s-domain near top edge: s ∈ [1-window, 1 - margin]
    smax = 1.0 - max(s_margin, 10*S_GUARD)
    smin = max(-0.999, smax - window_half)

    # Chebyshev clustered global grid and anchors (descending)
    sgrid   = cheb_nodes_desc(smin, smax, npts)
    anchors = cheb_nodes_desc(smin, smax, nanchors)

    @info "JUE (STRONG v2) N=$N, a=$a, b=$b; domain [$(@sprintf("%.6f",smin)),$(@sprintf("%.6f",smax))]; Nquad=$NquadFD; anchors=$nanchors"

    # corresponding t-grids
    tgrid    = 0.5 .* (sgrid   .+ 1.0)
    tanchors = 0.5 .* (anchors .+ 1.0)

    # local half_step around each anchor (robust spacing)
    function local_half_step(j)
        if j == 1
            return 0.5*abs(anchors[2]-anchors[1])
        elseif j == length(anchors)
            return 0.5*abs(anchors[end]-anchors[end-1])
        else
            return 0.25*abs(anchors[j+1]-anchors[j-1])
        end
    end

    # Build anchor data: (F0 from direct Fredholm) + (σ,σ' from LS)
    anc = Vector{NamedTuple}(undef, length(anchors))
    for (j, s0) in pairs(anchors)
        hstep = local_half_step(j)
        spts = safe_stencil_points(s0, hstep; lo=-1.0+S_GUARD, hi=1.0-S_GUARD)
        Fpts = F_N_vec_JUE(N, a, b, spts; Nquad=NquadFD)
        fit  = sigma_fit_logF(spts, Fpts)
        # slightly higher quadrature for anchor value
        F0_direct = F_N_fredholm_JUE(N, a, b, s0; Nquad=NquadFD+40)
        F0a = clamp(F0_direct, 0.0, 1.0)
        if F0a < 1e-300; F0a = 0.0; end
        anc[j] = (s0=s0, σ=fit.σ, σp=fit.σp, F0=F0a)
        (j % 10 == 0) && @info "  anchor $j/$(length(anchors)) at s0=$(round(s0,digits=6))"
    end

    # PVI parameters
    v1 = N + 0.5*(a + b)
    v2 = 0.5*(a + b)
    v4 = 0.5*(b - a)

    # Integrate IN t (descending), with branch tracking; relock at t-anchors
    fvals  = fill(NaN, length(sgrid))
    fpvals = fill(NaN, length(sgrid))

    # seed from first anchor
    F0 = anc[1].F0
    σ  = anc[1].σ
    σp = anc[1].σp
    fvals[1]  = F0
    fpvals[1] = 2.0 * σ * F0
    fpp_local = 4.0 * (σp + σ^2) * F0
    k = 1

    for j in 1:(length(anchors)-1)
        tR = tanchors[j+1]
        # march to the next anchor
        while k < length(tgrid) && tgrid[k] > tR + 1e-12
            fvals[k+1], fpvals[k+1], fpp_local =
                step_f_pvi_guarded!(fvals[k], fpvals[k], tgrid[k], tgrid[k+1],
                                    v1, v2, v4, fpp_local; cap_coeff=0.10)
            k += 1
        end
        # relock at the next anchor, enforcing direction-aware monotonicity with previous value
        idx   = findmin(abs.(tgrid .- tR))[2]
        prevF = fvals[k]  # last computed value before reset
        F0a   = anc[j+1].F0
        # moving left (decreasing t): F must not jump up
        F0a = min(F0a, prevF + ANCHOR_EPS)
        F0a = clamp(F0a, 0.0, 1.0)
        σ  = anc[j+1].σ
        σp = anc[j+1].σp
        fvals[idx]  = F0a
        fpvals[idx] = 2.0 * σ * fvals[idx]
        fpp_local   = 4.0 * (σp + σ^2) * fvals[idx]
        k = idx
    end

    # ---- Final march from the last anchor to the end of the grid ----
    while k < length(tgrid)
        fvals[k+1], fpvals[k+1], fpp_local =
            step_f_pvi_guarded!(fvals[k], fpvals[k], tgrid[k], tgrid[k+1],
                                v1, v2, v4, fpp_local; cap_coeff=0.10)
        k += 1
    end

    F_pvi = fvals                      # PVI solution f(t)=F(t)
    F_fd  = F_N_vec_JUE(N, a, b, sgrid; Nquad=max(NquadFD, 320))

    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-N Fredholm (Jacobi, Gram)",
                xlabel="s", ylabel="CDF",
                title="JUE largest-eigenvalue CDF (N=$(N), a=$(a), b=$(b))")
    plot!(plt1, sgrid, F_pvi, lw=2, ls=:dash, label="Painlevé VI (locked, t=(s+1)/2) — STRONG v2")
    savefig(plt1, "jue_pvi_locked_vs_fd_STRONG2_N$(N)_a$(round(a,digits=3))_b$(round(b,digits=3)).png"); display(plt1)

    plt2 = plot(sgrid, abs.(F_fd .- F_pvi), lw=2, label="|Δ|",
                xlabel="s", ylabel="absolute error",
                title="Absolute difference (STRONG v2): PVI vs Fredholm (N=$(N), a=$(a), b=$(b))")
    savefig(plt2, "jue_pvi_locked_absdiff_STRONG2_N$(N)_a$(round(a,digits=3))_b$(round(b,digits=3)).png"); display(plt2)

    @info "STRONG v2  N=$N, a=$a, b=$b: max |Δ| = $(maximum(abs.(F_fd .- F_pvi)))"
    return (sgrid=sgrid, F_fd=F_fd, F_pvi=F_pvi, plt1=plt1, plt2=plt2)
end

# ---------------- Run a small suite ----------------
function run_jue_suite_strong2()
    configs = [
        (N=20,  a=1.0, b=2.0, Nquad=300, npts=1401, anchors=161, window=0.25),
        (N=50,  a=2.0, b=3.0, Nquad=320, npts=1601, anchors=181, window=0.25),
        (N=100, a=5.0, b=1.0, Nquad=340, npts=1801, anchors=201, window=0.22),
    ]
    for c in configs
        jue_pvi_locked_strong2(N=c.N, a=c.a, b=c.b,
                               NquadFD=c.Nquad, npts=c.npts, nanchors=c.anchors,
                               window_half=c.window)
    end
    @info "Saved all figures in $(pwd())"
end

# Define everything first, then call:
run_jue_suite_strong2()
