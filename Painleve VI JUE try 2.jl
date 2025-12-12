# finite_n_pvi_jue_locked_suite_ultrafast.jl
# JUE verifier: caches GL nodes, precomputes Jacobi norms, adaptive Nquad(s),
# low-rank Gram determinant, threaded Fredholm, y-integration for PVI.

import Pkg
for p in ["FastGaussQuadrature","LinearAlgebra","Statistics","Plots","SpecialFunctions","Printf"]
    Base.find_package(p) === nothing && (try Pkg.add(p) catch; Pkg.Registry.update(); Pkg.add(p); end)
end

using FastGaussQuadrature, LinearAlgebra, Statistics, Plots, SpecialFunctions
using Printf: @sprintf
using Base.Threads: @threads, nthreads
gr()

const S_GUARD    = 1e-10
const ANCHOR_EPS = 1e-12
const F_FLOOR    = 0.0
const F_CEIL     = 1.0

# ---------------- Utilities ----------------
cheb_nodes_asc(a::Float64, b::Float64, m::Int) = begin
    xs = [cos((2k-1)*pi/(2m)) for k in 1:m]   # (-1,1)
    sort!( @. 0.5*((b-a)*xs + (b+a)) )        # [a,b], ascending
end

# Cache for GL nodes
const GL_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
get_gl(Nq::Int) = get!(GL_CACHE, Nq) do; gausslegendre(Nq) end

# ---------------- Jacobi orthonormal system ----------------
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
    v
end

function jacobi_phi_row!(row::AbstractVector{Float64}, x::Float64,
                         N::Int, a::Float64, b::Float64, invsh::Vector{Float64})
    @assert length(row) == N
    if !(x > -1.0 && x < 1.0)
        error("Jacobi support is (-1,1). Got x=$x")
    end
    sqrtw = (1 - x)^(0.5*a) * (1 + x)^(0.5*b)

    # P0, P1
    Pkm1 = 1.0
    row[1] = Pkm1 * invsh[1] * sqrtw
    if N == 1; return; end
    Pk = 0.5*((a - b) + (a + b + 2.0)*x)
    row[2] = Pk * invsh[2] * sqrtw
    if N == 2; return; end

    # 3-term recurrence (DLMF 18.9.5)
    for n in 1:(N-2)  # produces P_{n+1}
        c1 = 2.0*(n+1.0)*(n + a + b + 1.0)*(2.0*n + a + b)
        c2 = (2.0*n + a + b + 1.0)*((2.0*n + a + b + 2.0)*(2.0*n + a + b)*x + a^2 - b^2)
        c3 = 2.0*(n + a)*(n + b)*(2.0*n + a + b + 2.0)
        Pkp1 = (c2*Pk - c3*Pkm1)/c1
        Pkm1, Pk = Pk, Pkp1
        row[n+2] = Pk * invsh[n+2] * sqrtw
    end
end

function Phi_jacobi!(Φ::Matrix{Float64}, t::Vector{Float64},
                     N::Int, a::Float64, b::Float64, invsh::Vector{Float64})
    @inbounds for i in eachindex(t)
        jacobi_phi_row!(view(Φ, i, :), t[i], N, a, b, invsh)
    end
    Φ
end

# ---------------- Fast Fredholm (low-rank Gram) with caching & adaptive Nquad(s) ----------------
@inline nquad_for_s(s::Float64; Nmin::Int=180, Nmax::Int=540, C::Float64=75.0) = begin
    δ = max(5e-5, 1.0 - s)                 # larger δ → fewer nodes
    Nmin + min(Nmax - Nmin, Int(ceil(C / sqrt(δ))))
end

function fredholm_JUE(N::Int, a::Float64, b::Float64, s::Float64,
                      invsh::Vector{Float64}; Nmin::Int=180, Nmax::Int=540, C::Float64=75.0)
    s_eff = min(max(s, -1.0 + S_GUARD), 1.0 - S_GUARD)
    s_eff <= -1 || s_eff >= 1 && error("s must lie in (-1,1). Got s=$(s)")
    Nq = nquad_for_s(s_eff; Nmin=Nmin, Nmax=Nmax, C=C)
    z, w = get_gl(Nq)                                  # cached nodes
    t  = 0.5 .* ((1.0 - s_eff).*z .+ (1.0 + s_eff))   # (s,1)
    dt = 0.5 .* (1.0 - s_eff) .* w

    Φ  = Matrix{Float64}(undef, Nq, N)                 # build once per s
    Phi_jacobi!(Φ, t, N, a, b, invsh)
    @inbounds for i in 1:Nq
        @inbounds @fastmath Φ[i, :] .*= sqrt(dt[i])    # WΦ (row-scale)
    end
    G = transpose(Φ) * Φ                                # N×N Gram  (BLAS-3 fast)
    G = 0.5 .* (G + transpose(G))
    val, _ = logabsdet(I + (-1.0).*G)                  # log|det(I - G)|
    return exp(val)
end

function fredholm_vec_JUE(N::Int, a::Float64, b::Float64, svals::AbstractVector{<:Real},
                          invsh::Vector{Float64};
                          Nmin::Int=180, Nmax::Int=540, C::Float64=75.0,
                          use_threads::Bool=true)
    F = Vector{Float64}(undef, length(svals))
    if use_threads
        @threads for i in eachindex(svals)
            F[i] = fredholm_JUE(N, a, b, Float64(svals[i]), invsh; Nmin=Nmin, Nmax=Nmax, C=C)
        end
    else
        @inbounds for i in eachindex(svals)
            F[i] = fredholm_JUE(N, a, b, Float64(svals[i]), invsh; Nmin=Nmin, Nmax=Nmax, C=C)
        end
    end
    F
end

# ---------------- σ,σ′ fit on log F + smoothing ----------------
function sigma_fit_logF(spts::Vector{Float64}, Fvals::Vector{Float64})
    s0 = spts[cld(length(spts),2)]
    u  = spts .- s0
    y  = log.(Fvals)
    M  = hcat(u.^0, u, u.^2, u.^3, u.^4)    # degree-4

    du   = sort(abs.(diff(sort(spts))))
    hmed = isempty(du) ? 1e-3 : max(median(du), 1e-6)
    σw   = 2.0*hmed
    w    = exp.(-(u./σw).^2)
    W    = Diagonal(w)

    c = (M' * W * M) \ (M' * W * y)
    σ  = c[2]
    σp = 2c[3]
    return (; s0, σ, σp)
end

function smooth5!(arr::Vector{Float64})
    n = length(arr); n < 5 && return arr
    ker = [1.0, 2.0, 3.0, 2.0, 1.0]; s = sum(ker)
    buf = copy(arr)
    @inbounds for i in 1:n
        i1 = clamp(i-2, 1, n); i2 = clamp(i-1, 1, n); i3 = i
        i4 = clamp(i+1, 1, n); i5 = clamp(i+2, 1, n)
        buf[i] = (ker[1]*arr[i1] + ker[2]*arr[i2] + ker[3]*arr[i3] +
                  ker[4]*arr[i4] + ker[5]*arr[i5]) / s
    end
    copyto!(arr, buf); arr
end

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

# ---------------- Painlevé VI algebraic relation & y-dynamics ----------------
@inline function pvi_rhs_sq(t::Float64, f::Float64, fp::Float64, v1::Float64, v2::Float64, v4::Float64)
    term1 = 4.0*t*(1.0-t)*(fp^3)
    term2 = 4.0*(1.0 - 2.0*t)*(fp^2)*f
    term3 = 4.0*fp*(f^2)
    term4 = -4.0*(f^2)*(v1^2)
    term5 = (fp^2) * ( 4.0*t*(v1^2)/(1.0 - t) - ( (v2 - v4)^2 )/4.0 - 4.0*t*v1*v4 )
    term6 = 4.0*f*fp*( -v1^2 + 2.0*t*v1^2 + v1*v4 )
    return term1 - term2 - term3 + term4 - term5 - term6
end

@inline function choose_fpp_t(prev_fpp::Float64, t::Float64, f::Float64, fp::Float64,
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

# y = -log(1 - t):  ṫ = 1 - t,  ḟ = g,  ġ = -g + (1 - t)^2 f''(t)
@inline function y_rhs(f::Float64, g::Float64, t::Float64, prev_fpp_t::Float64,
                       v1::Float64, v2::Float64, v4::Float64)
    one_minus_t = max(1e-14, 1.0 - t)
    fp = g / one_minus_t
    fpp_t = choose_fpp_t(prev_fpp_t, t, f, fp, v1, v2, v4)
    fdot = g
    tdot = one_minus_t
    gdot = -g + (one_minus_t^2) * fpp_t
    return fdot, gdot, tdot, fpp_t
end

mutable struct RKState
    f::Float64; g::Float64; t::Float64; fpp_t::Float64
end

function rk32_step!(st::RKState, h::Float64, v1::Float64, v2::Float64, v4::Float64)
    f1, g1, t1, fpp1 = y_rhs(st.f, st.g, st.t, st.fpp_t, v1, v2, v4)

    f2, g2, t2, fpp2 = y_rhs(st.f + 0.5*h*f1,
                              st.g + 0.5*h*g1,
                              st.t + 0.5*h*t1,
                              fpp1, v1, v2, v4)

    f3, g3, t3, fpp3 = y_rhs(st.f + 0.75*h*f2,
                              st.g + 0.75*h*g2,
                              st.t + 0.75*h*t2,
                              fpp2, v1, v2, v4)

    f_new  = st.f + h*(2/9*f1 + 1/3*f2 + 4/9*f3)
    g_new  = st.g + h*(2/9*g1 + 1/3*g2 + 4/9*g3)
    t_new  = st.t + h*(2/9*t1 + 1/3*t2 + 4/9*t3)

    f2e = st.f + h*( 7/24*f1 + 1/4*f2 + 1/3*f3 )
    g2e = st.g + h*( 7/24*g1 + 1/4*g2 + 1/3*g3 )

    err_f = abs(f_new - f2e)
    err_g = abs(g_new - g2e)

    fpp_new = fpp3
    return f_new, g_new, t_new, fpp_new, err_f, err_g
end

function integrate_y!(st::RKState, y0::Float64, y1::Float64;
                      v1::Float64, v2::Float64, v4::Float64,
                      atol_f::Float64=5e-7, rtol_f::Float64=5e-6,
                      atol_g::Float64=5e-7, rtol_g::Float64=5e-6,
                      h_init::Float64=0.05, h_min::Float64=1e-5)

    y = y0
    h = min(h_init, max(h_min, (y1 - y0)/4))
    f_prev = st.f

    while y < y1 - 1e-14
        if y + h > y1; h = y1 - y; end
        f_try, g_try, t_try, fpp_try, e_f, e_g = rk32_step!(st, h, v1, v2, v4)

        sf = atol_f + rtol_f*max(abs(st.f), abs(f_try))
        sg = atol_g + rtol_g*max(abs(st.g), abs(g_try))
        err = max(e_f / sf, e_g / sg)

        if err <= 1.0
            if f_try < f_prev
                h = max(h_min, 0.5*h);  continue
            end
            f_try = clamp(f_try, F_FLOOR, F_CEIL)
            t_try = min(max(st.t, t_try), 1.0 - 1e-12)
            st.f, st.g, st.t, st.fpp_t = f_try, g_try, t_try, fpp_try
            y += h
            f_prev = st.f
            h = min(0.5*(y1 - y) + h, max(1.1*h, h * max(0.2, 0.9 * err^(-1/3))))
        else
            h = max(h_min, h * max(0.2, 0.9 * err^(-1/3)))
        end
    end
end

# ---------------- One full run ----------------
function jue_pvi_locked_ultrafast(; N::Int=50, a::Float64=0.0, b::Float64=0.0,
                                   npts::Int=901, nanchors::Int=101,
                                   s_margin::Float64=5e-6, window_half::Float64=0.25,
                                   Nmin::Int=180, Nmax::Int=540, Cgrow::Float64=75.0)

    # domain
    smax = 1.0 - max(s_margin, 10*S_GUARD)
    smin = max(-0.999, smax - window_half)
    sgrid   = cheb_nodes_asc(smin, smax, npts)
    anchors = cheb_nodes_asc(smin, smax, nanchors)

    @info "JUE (ULTRA) N=$N, a=$a, b=$b; domain [$(@sprintf("%.6f",smin)),$(@sprintf("%.6f",smax))]; "*
          "anchors=$nanchors; threads=$(nthreads())"

    # precompute Jacobi norms ONCE
    invsh = inv_sqrt_h_vec(N, a, b)

    # t and y
    tgrid    = 0.5 .* (sgrid   .+ 1.0)
    tanchors = 0.5 .* (anchors .+ 1.0)
    ygrid    = -log.(1 .- tgrid)
    yanchors = -log.(1 .- tanchors)

    # local half-step for LS stencils
    function local_half_step(j)
        if j == 1
            return 0.5*abs(anchors[2]-anchors[1])
        elseif j == length(anchors)
            return 0.5*abs(anchors[end]-anchors[end-1])
        else
            return 0.25*abs(anchors[j+1]-anchors[j-1])
        end
    end

    # Anchors: σ,σ′,F0
    rawσ  = zeros(Float64, length(anchors))
    rawσp = zeros(Float64, length(anchors))
    F0s   = zeros(Float64, length(anchors))

    for (j, s0) in pairs(anchors)
        hstep = local_half_step(j)
        spts = safe_stencil_points(s0, hstep; lo=-1.0+S_GUARD, hi=1.0-S_GUARD)
        Fpts = fredholm_vec_JUE(N, a, b, spts, invsh; Nmin=Nmin, Nmax=Nmax, C=Cgrow, use_threads=true)
        fit  = sigma_fit_logF(spts, Fpts)
        F0_direct = fredholm_JUE(N, a, b, s0, invsh; Nmin=Nmin, Nmax=Nmax, C=Cgrow)
        F0a = clamp(F0_direct, F_FLOOR, F_CEIL)
        if F0a < 1e-300; F0a = 0.0; end
        rawσ[j]  = fit.σ
        rawσp[j] = fit.σp
        F0s[j]   = F0a
        (j % 15 == 0) && @info "  anchor $j/$(length(anchors)) at s0=$(round(s0,digits=6))"
    end

    σs  = copy(rawσ);  smooth5!(σs)
    σps = copy(rawσp); smooth5!(σps)

    # PVI parameters
    v1 = N + 0.5*(a + b)
    v2 = 0.5*(a + b)
    v4 = 0.5*(b - a)

    # Outputs
    F_pvi = fill(NaN, length(sgrid))

    # Seed at first anchor
    ia = 1
    y  = yanchors[ia]
    t0 = tanchors[ia]
    f0 = F0s[ia]
    σ0 = σs[ia]; σp0 = σps[ia]
    st = RKState(
        f0,
        (1 - t0) * (2.0 * σ0 * f0),     # g = (1-t) f'
        t0,
        4.0 * (σp0 + σ0^2) * f0
    )

    # Record if starting on a grid node
    ig = searchsortedfirst(ygrid, yanchors[ia])
    if ig <= length(ygrid) && abs(yanchors[ia] - ygrid[ig]) ≤ 1e-12
        F_pvi[ig] = clamp(st.f, F_FLOOR, F_CEIL)
        ig += 1
    end

    # Event loop (grid ∪ anchors)
    while ig <= length(ygrid)
        y_next_grid   = ygrid[ig]
        y_next_anchor = ia < length(yanchors) ? yanchors[ia+1] : Inf
        y_next = min(y_next_grid, y_next_anchor)

        integrate_y!(st, y, y_next; v1=v1, v2=v2, v4=v4,
                     atol_f=5e-7, rtol_f=5e-6, atol_g=5e-7, rtol_g=5e-6,
                     h_init=0.05, h_min=1e-5)
        y = y_next

        if abs(y - y_next_grid) ≤ 1e-12
            F_pvi[ig] = clamp(st.f, F_FLOOR, F_CEIL)
            ig += 1
            continue
        end

        if ia < length(yanchors) && abs(y - yanchors[ia+1]) ≤ 1e-12
            ia += 1
            F0a = max(F0s[ia], st.f - ANCHOR_EPS)     # monotone in y
            F0a = clamp(F0a, F_FLOOR, F_CEIL)
            σa, σpa = σs[ia], σps[ia]
            t_a  = 1 - exp(-yanchors[ia])
            st.f = F0a
            st.t = t_a
            st.g = (1 - st.t) * (2.0 * σa * st.f)
            st.fpp_t = 4.0 * (σpa + σa^2) * st.f
        end
    end

    # Two-pass fill for any NaNs
    j0 = findfirst(!isnan, F_pvi)
    if j0 === nothing
        error("PVI produced no samples; check anchors/grid alignment.")
    end
    for i in 1:j0-1
        F_pvi[i] = F_pvi[j0]
    end
    for i in (j0+1):length(F_pvi)
        if isnan(F_pvi[i]); F_pvi[i] = F_pvi[i-1]; end
    end

    # Fredholm on same grid (fast, adaptive, cached)
    F_fd = fredholm_vec_JUE(N, a, b, sgrid, invsh; Nmin=Nmin, Nmax=Nmax, C=Cgrow, use_threads=true)

    # Plots
    plt1 = plot(sgrid, F_fd, lw=2, label="Finite-N Fredholm (low-rank, cached, adaptive)",
                xlabel="s", ylabel="CDF",
                title="JUE largest-eigenvalue CDF (N=$(N), a=$(a), b=$(b)) — ULTRA (threads=$(nthreads()))")
    plot!(plt1, sgrid, F_pvi, lw=2, ls=:dash, label="Painlevé VI (locked, y-integration)")
    savefig(plt1, "jue_pvi_locked_vs_fd_ULTRA_N$(N)_a$(round(a,digits=3))_b$(round(b,digits=3)).png"); display(plt1)

    diff = abs.(F_fd .- F_pvi)
    plt2 = plot(sgrid, diff, lw=2, label="|Δ|",
                xlabel="s", ylabel="absolute error",
                title="Absolute difference (ULTRA): PVI vs Fredholm (N=$(N), a=$(a), b=$(b))")
    savefig(plt2, "jue_pvi_locked_absdiff_ULTRA_N$(N)_a$(round(a,digits=3))_b$(round(b,digits=3)).png"); display(plt2)

    @info "ULTRA N=$N, a=$a, b=$b: max |Δ| = $(maximum(diff))"
    return (sgrid=sgrid, F_fd=F_fd, F_pvi=F_pvi, plt1=plt1, plt2=plt2)
end

# ---------------- Quick suite ----------------
function run_jue_suite_ultrafast()
    configs = [
        (N=20,  a=1.0, b=2.0, npts=801, anchors=91,  window=0.25),
        (N=50,  a=2.0, b=3.0, npts=901, anchors=101, window=0.25),
        (N=100, a=5.0, b=1.0, npts=1001,anchors=111, window=0.22),
    ]
    for c in configs
        jue_pvi_locked_ultrafast(N=c.N, a=c.a, b=c.b,
                                 npts=c.npts, nanchors=c.anchors,
                                 window_half=c.window,
                                 Nmin=180, Nmax=540, Cgrow=75.0)
    end
    @info "Saved all figures in $(pwd())"
end

# Run on include()
run_jue_suite_ultrafast()
