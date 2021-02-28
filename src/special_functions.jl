module SpecialFunctions
using OffsetArrays
using ..Constants

export atcdim, ricbessel, richankel, cricbessel, crichankel, cspherebessel, vcfunc, normalizedlegendre

const ci = 1.0im

"""
    ricbessel(n, ds, ϵ, Ψ)

Compute the Ricatti-Bessel function.

References:

- [Riccati-Bessel Functions](https://mathworld.wolfram.com/Riccati-BesselFunctions.html) at Wolfram MathWorld
"""
function ricbessel(n::Int64, ds::Float64, ϵ::Float64)
    Ψ = OffsetArray(zeros(n + 1), 0:n)

    if ds < n
        ns = floor(Int, ds + 4(ds^0.3333) + 17)
        ns = max(n + 10, ns)
        dns = 0
        for i in (ns - 1):-1:n
            sn = (i + 1) / ds
            dns = sn - 1 / (dns + sn)
        end
        Ψ[n] = dns
        Ψ[n - 1] = n / ds - 1 / (dns + (n) / ds)
        for i in (n - 2):-1:1
            sn = (i + 1) / ds
            Ψ[i] = sn - 1 / (Ψ[i + 1] + sn)
        end
        Ψ_t = sin(ds)
        Ψ[0] = Ψ_t
        ds2 = ds * ds
        sum = Ψ_t * Ψ_t / ds2
        for i in 1:n
            Ψ_t = Ψ_t / (i / ds + Ψ[i])
            sum = sum + (2i + 1) * Ψ_t * Ψ_t / ds2
            err = abs(1 - sum)
            Ψ[i] = Ψ_t
            if err < ϵ
                # nmax = i
                return Ψ
            end
        end
        # nmax = n
    else
        Ψ[0] = sin(ds)
        Ψ[1] = Ψ[0] / ds - cos(ds)
        for i in 1:(n - 1)
            sn = (2i + 1) / ds
            Ψ[i + 1] = sn * Ψ[i] - Ψ[i - 1]
        end
        # nmax = n
    end

    return Ψ
end

function richankel(n::Int64, ds::Float64)
    Ξ = OffsetArray(zeros(ComplexF64, n + 1), 0:n)

    if ds < n
        ns = floor(Int, ds + 4(ds^0.3333) + 17)
        ns = max(n + 10, ns)
        dns = 0
        for i in (ns - 1):-1:n
            sn = (i + 1) / ds
            dns = sn - 1 / (dns + sn)
        end
        Ξ[n] = dns
        Ξ[n - 1] = (n) / ds - 1 / (dns + (n) / ds)
        for i in (n - 2):-1:1
            sn = (i + 1) / ds
            Ξ[i] = sn - 1 / (Ξ[i + 1] + sn)
        end
        chi0 = -cos(ds)
        psi = sin(ds)
        chi1 = chi0 / ds - psi
        Ξ[0] = ComplexF64(psi, chi0)
        for i in 1:n
            chi2 = (2i + 1) / ds * chi1 - chi0
            psi = psi / ((i) / ds + Ξ[i])
            Ξ[i] = ComplexF64(psi, chi1)
            chi0 = chi1
            chi1 = chi2
        end
    else
        chi0 = -cos(ds)
        psi0 = sin(ds)
        chi1 = chi0 / ds - psi0
        psi1 = psi0 / ds + chi0
        Ξ[0] = ComplexF64(psi0, chi0)
        Ξ[1] = ComplexF64(psi1, chi1)
        for i in 1:(n - 1)
            sn = (2i + 1) / ds
            Ξ[i + 1] = sn * Ξ[i] - Ξ[i - 1]
        end
    end
    return Ξ
end

"""
    cricbessel(n, ds)

Compute the Riccati-Bessel function, return complex values.
"""
function cricbessel(n::Int64, ds::ComplexF64)
    Ψ, Χ = cspherebessel(n, ds)
    for i in 0:n
        Ψ[i] *= ds
    end
    return Ψ
end

function crichankel(n::Int64, ds::ComplexF64)
    Ξ = OffsetArray(zeros(ComplexF64, n + 1), 0:n)
    Ξ[0] = -ci * exp(ci * ds)
    Ψ_0 = sin(ds)
    if abs(Ξ[0]) / abs(Ψ_0) < 1e-6
        Ξ[1] = -exp(ci * ds) * (ci + ds) / ds
        for i in 2:n
            Ξ[i] = (2i + 1) / ds * Ξ[i - 1] - Ξ[i - 2]
        end
    else
        Ψ, Χ = cspherebessel(n, ds)
        for i in 1:n
            Ξ[i] = (Ψ[i] + ci * Χ[i]) * ds
        end
    end
    return Ξ
end

"""
    cspherebessel(n, z)

Compute spherical Bessel functions jn(z) & yn(z)
"""
function cspherebessel(n::Int64, z::ComplexF64)
    a0 = abs(z)
    nm = n
    csj = OffsetArray(zeros(ComplexF64, n + 1), 0:n)
    csy = OffsetArray(zeros(ComplexF64, n + 1), 0:n)
    if a0 < 1e-60
        fill!(csy, ComplexF64(-1e300, 0))
        csy[0] = ComplexF64(1, 0)
        return csj, csy
    end
    csj[0] = sin(z) / z
    csj[1] = (csj[0] - cos(z)) / z
    if n >= 2
        csa = csj[0]
        csb = csj[1]
        m = msta1(a0, 200)
        if m < n
            nm = m
        else
            m = msta2(a0, n, 15)
        end
        cf = 0.0
        cf0 = 0.0
        cf1 = 1.0 - 100
        for k in m:-1:0
            cf = (2.0 * k + 3.0) * cf1 / z - cf0
            if k <= nm
                csj[k] = cf
            end
            cf0 = cf1
            cf1 = cf
        end
        if (abs(csa) > abs(csb))
            cs = csa / cf
        else
            cs = csb / cf0
        end
        for k in 0:min(nm, n)
            csj[k] = cs * csj[k]
        end
    end
    fill!(csy, ComplexF64(1e200, 0))
    csy[0] = -cos(z) / z
    csy[1] = (csy[0] - sin(z)) / z
    for k in 2:min(nm, n)
        if abs(csj[k - 1]) > abs(csj[k - 2])
            csy[k] = (csj[k] * csy[k - 1] - 1.0 / (z * z)) / csj[k - 1]
        else
            csy[k] = (csj[k] * csy[k - 2] - (2.0 * k - 1.0) / z^3) / csj[k - 2]
        end
    end

    return csj, csy
end

"""
    msta1(x, mp)

Determine the starting point for backward recurrence such that the magnitude of Jn(x) at that point is about 10^(-MP).
"""
function msta1(x::Float64, mp::Int64)
    a0 = abs(x)
    n0 = floor(Int, 1.1 * a0) + 1
    f0 = envj(n0, a0) - mp
    n1 = n0 + 5
    f1 = envj(n1, a0) - mp
    nn = 0
    for it in 1:20
        nn = floor(Int, n1 - (n1 - n0) / (1 - f0 / f1))
        f = envj(nn, a0) - mp
        if abs(nn - n1) < 1
            break
        end
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    end
    return nn
end

function msta2(x::Float64, n::Int64, mp::Int64)
    a0 = abs(x)
    hmp = 0.5mp
    ejn = envj(n, a0)
    if ejn <= hmp
        obj = mp
        n0 = floor(Int, 1.1 * a0)
    else
        obj = hmp + ejn
        n0 = n
    end
    f0 = envj(n0, a0) - obj
    n1 = n0 + 5
    f1 = envj(n1, a0) - obj
    nn = 0
    for it in 1:20
        nn = floor(Int, n1 - (n1 - n0) / (1.0 - f0 / f1))
        f = envj(nn, a0) - obj
        if abs(nn - n1) < 1
            break
        end
        n0 = n1
        f0 = f1
        n1 = nn
        f1 = f
    end
    return nn + 10
end

function envj(n::Int64, x::Float64)
    n = max(1, abs(n))
    return 0.5log10(6.28n) - n * log10(1.36 * x / n)
end

function vcfunc(ctx::ConstantContext, m::Int64, n::Int64, k::Int64, l::Int64)
    vcn = OffsetArray(zeros(n + l + 1), 0:(n + l))
    vcfunc!(ctx, m, n, k, l, vcn)
    return vcn
end

function vcfunc!(ctx::ConstantContext, m::Int64, n::Int64, k::Int64, l::Int64, vcn::OffsetArray{Float64,1})
    init!(ctx, n + l)

    bcof, fnr, _, _ = get_offset_constants(ctx)
    wmax = n + l
    wmin = max(abs(n - l), abs(m + k))
    fill!(vcn, 0)
    vcn[wmax] = bcof[n + m, l + k] * bcof[n - m, l - k] / bcof[n + n, l + l]
    if wmin == wmax
        return
    end
    vcn[wmax - 1] =
        vcn[wmax] * (l * m - k * n) * fnr[2(l + n) - 1] / fnr[l] / fnr[n] / fnr[n + l + m + k] / fnr[n + l - m - k]
    if wmin == wmax - 1
        return
    end
    mk = m + k
    vcmax = abs(vcn[wmax]) + abs(vcn[wmax - 1])

    # a downwards recurrence is used initially
    for w in wmax:-1:(wmin + 2)
        t1::Float64 =
            2w * fnr[2w + 1] * fnr[2w - 1] /
            (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        t3::Float64 =
            fnr[w - mk - 1] *
            fnr[w + mk - 1] *
            fnr[l - n + w - 1] *
            fnr[n - l + w - 1] *
            fnr[n + l - w + 2] *
            fnr[n + l + w] / ((2(w - 1)) * fnr[2w - 3] * fnr[2w - 1])
        vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3
        if (wmax - w) % 2 == 1
            vctest = abs(vcn[w - 2]) + abs(vcn[w - 1])
            vcmax = max(vcmax, vctest)
            rat = vctest / vcmax
            # if/when the coefficients start to decrease in magnitude, an upwards recurrence takes over
            if rat < 0.01 && w - 2 > wmin
                wmax = w - 3
                vcfuncuprec!(ctx, m, n, k, l, wmax, vcn)
                break
            end
        end
    end
end

"""
    vcfuncuprec(ctx, m, n, k, l, wmax, vcn)

Upwards VC coefficient recurrence, used internally by vcfunc(ctx, m, n, k, l).
"""
function vcfuncuprec!(
    ctx::ConstantContext,
    m::Int64,
    n::Int64,
    k::Int64,
    l::Int64,
    wmax::Int64,
    vcn::OffsetArray{Float64,1},
)
    bcof, fnr, monen, _ = get_offset_constants(ctx)

    mk = abs(m + k)
    nl = abs(n - l)
    if nl >= mk
        w = nl
        if n >= l
            m1 = m
            n1 = n
            l1 = l
            k1 = k
        else
            m1 = k
            n1 = l
            k1 = m
            l1 = n
        end
        vc1 = monen[k1 + l1] * bcof[l1 + k1, w - m1 - k1] * bcof[l1 - k1, w + m1 + k1] / bcof[2l1, 2w + 1]
    else
        w = mk
        if m + k >= 0
            vc1 = monen[n + m] * bcof[n - l + w, l - k] * bcof[l - n + w, n - m] / bcof[2w + 1, n + l - w]
        else
            vc1 = monen[l + k] * bcof[n - l + w, l + k] * bcof[l - n + w, n + m] / bcof[2w + 1, n + l - w]
        end
    end

    w1 = w
    vcn[w] = vc1
    w = w1 + 1
    mk = m + k
    w2 = min(wmax, n + l)
    if w2 > w1
        t1 =
            2 * w * fnr[2w + 1] * fnr[2w - 1] /
            (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        if w1 == 0
            t2 = 0.5 * (m - k)
        else
            t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        end
        vcn[w] = t1 * t2 * vcn[w1]
    end

    for w in (w1 + 2):w2
        t1 =
            2w * fnr[2w + 1] * fnr[2w - 1] /
            (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        t3 =
            fnr[w - mk - 1] *
            fnr[w + mk - 1] *
            fnr[l - n + w - 1] *
            fnr[n - l + w - 1] *
            fnr[n + l - w + 2] *
            fnr[n + l + w] / ((2 * (w - 1)) * fnr[2w - 3] * fnr[2w - 1])
        vcn[w] = t1 * (t2 * vcn[w - 1] - t3 * vcn[w - 2])
    end

    return
end

function normalizedlegendre(ctx::ConstantContext, cbe::Float64, mmax::Int64, nmax::Int64)
    init!(ctx, mmax + nmax)

    bcof, fnr, monen, _ = get_offset_constants(ctx)
    dc = OffsetArray(zeros(2mmax + 1, nmax + 1), (-mmax):mmax, 0:nmax)

    # cbe must ∈ [-1, 1] otherwise sbe will be a complex number
    sbe = √((1 + cbe) * (1 - cbe))
    for m in 0:mmax
        dc[m, m] = monen[m] * (0.5 * sbe)^m * bcof[m, m]
        if m == nmax
            break
        end
        dc[m, m + 1] = fnr[2m + 1] * cbe * dc[m, m]
        for n in (m + 1):(nmax - 1)
            dc[m, n + 1] =
                (-fnr[n - m] * fnr[n + m] * dc[m, n - 1] + (2n + 1) * cbe * dc[m, n]) /
                (fnr[n + 1 - m] * fnr[n + 1 + m])
        end
    end

    for m in 1:mmax
        for n in m:nmax
            dc[-m, n] = monen[m] * dc[m, n]
        end
    end

    return dc
end

function rotcoef(ctx::ConstantContext, cbe::Float64, kmax::Int64, nmax::Int64)
    init!(ctx, kmax + nmax)
    bcof, fnr, monen, _ = get_offset_constants(ctx)

    dc = OffsetArray(zeros(2kmax + 1, nmax * (nmax + 2) + 1), (-kmax):kmax, 0:(nmax * (nmax + 2)))
    dk0 = OffsetArray(zeros(2nmax + 3), (-nmax - 1):(nmax + 1))
    dk01 = OffsetArray(zeros(2nmax + 3), (-nmax - 1):(nmax + 1))

    # cbe must ∈ [-1, 1] otherwise sbe will be a complex number
    sbe = √((1 + cbe) * (1 - cbe))
    cbe2 = 0.5(1 + cbe)
    sbe2 = 0.5(1 - cbe)

    dk0[0] = 1
    sben = 1.0
    dc[0, 0] = 1
    dk01[0] = 0

    for n in 1:nmax
        knmax = min(n, kmax)
        nn1 = n * (n + 1)
        sben *= sbe / 2
        dk0[n] = monen[n] * sben * bcof[n, n]
        dk0[-n] = monen[n] * dk0[n]
        dk01[n] = 0
        dk01[-n] = 0
        dc[0, nn1 + n] = dk0[n]
        dc[0, nn1 - n] = dk0[-n]

        for k in (-n + 1):(n - 1)
            kn = nn1 + k
            dkt = dk01[k]
            dk01[k] = dk0[k]
            dk0[k] = (cbe * (2n - 1) * dk01[k] - fnr[n - k - 1] * fnr[n + k - 1] * dkt) / (fnr[n + k] * fnr[n - k])
            dc[0, kn] = dk0[k]
        end

        for m in 1:knmax
            fmn = 1 / fnr[n - m + 1] / fnr[n + m]
            m1 = m - 1
            dkm0 = 0.0

            for k in (-n):n
                kn = nn1 + k
                dkm1 = dkm0
                dkm0 = dc[m1, kn]
                dkn1 = k == n ? 0.0 : dc[m1, kn + 1]
                dc[m, kn] =
                    (
                        fnr[n + k] * fnr[n - k + 1] * cbe2 * dkm1 - fnr[n - k] * fnr[n + k + 1] * sbe2 * dkn1 -
                        k * sbe * dc[m1, kn]
                    ) * fmn
                dc[-m, nn1 - k] = dc[m, kn] * monen[k + m]
            end
        end
    end

    return dc
end

"""
    taufunc(ctx, cb, nmax)

Normalized vector spherical harmonic function.
"""
function taufunc(ctx::ConstantContext, cb::Float64, nmax::Int64)
    drot = rotcoef(ctx, cb, 1, nmax)
    τ = OffsetArray(zeros(nmax + 2, nmax, 2), 0:(nmax + 1), 1:nmax, 1:2)
    for n in 1:nmax
        nn1 = n * (n + 1)
        fnm = √((2n + 1) / 2) / 4

        for m in (-n):-1
            mn = nn1 + m
            τ[n + 1, -m, 1] = -fnm * (-drot[-1, mn] + drot[1, mn])
            τ[n + 1, -m, 2] = -fnm * (drot[-1, mn] + drot[1, mn])
        end

        for m in 0:n
            mn = nn1 + m
            τ[m, n, 1] = -fnm * (-drot[-1, mn] + drot[1, mn])
            τ[m, n, 2] = -fnm * (drot[-1, mn] + drot[1, mn])
        end
    end
    return τ
end

"""
    pifunc(cb, ephi, nmax, ndim)

Vector spherical harmonic function.

TODO: `ndim` seems to be useless. Can it be safely deleted?
"""
function pifunc(ctx::ConstantContext, cb::Float64, ephi::ComplexF64, nmax::Int64, ndim::Int64)
    drot = rotcoef(ctx, cb, 1, nmax)

    _, fnr, _, _ = get_offset_constants(ctx)

    ephim = OffsetArray(zeros(ComplexF64, 2nmax + 1), (-nmax):nmax)
    ephim[0] = 1.0

    # ndim should be as large as or larger than nmax
    π_vec = OffsetArray(zeros(ComplexF64, ndim + 2, ndim, 2), 0:(ndim + 1), 1:ndim, 1:2)

    for m in 1:nmax
        ephim[m] = ephi * ephim[m - 1]
        ephim[-m] = conj(ephim[m])
    end

    for n in 1:nmax
        cin = (-1im)^(n) * fnr[2n + 1]
        nn1 = n * (n + 1)
        for m in n:-1:1
            π_vec[n + 1, m, 1] = cin * drot[1, nn1 - m] * ephim[-m]
            π_vec[n + 1, m, 2] = cin * drot[-1, nn1 - m] * ephim[-m]
        end

        for m in 0:n
            π_vec[m, n, 1] = cin * drot[1, nn1 + m] * ephim[m]
            π_vec[m, n, 2] = cin * drot[-1, nn1 + m] * ephim[m]
        end
    end

    return π_vec
end

function planewavecoef(ctx::ConstantContext, α::Float64, β::Float64, nodr::Int64)
    init!(ctx, nodr)

    _, _, monen, _ = get_offset_constants(ctx)

    τ_lr = OffsetArray(zeros(nodr + 2, nodr, 2), 0:(nodr + 1), 1:nodr, 1:2)
    pmnp0 = OffsetArray(zeros(ComplexF64, nodr + 2, nodr, 2, 2), 0:(nodr + 1), 1:nodr, 1:2, 1:2)

    cb = cos(β)
    sb = √((1 - cb) * (1 + cb))
    ca = cos(α)
    sa = sin(α)
    E_α = ComplexF64(ca, sa)
    τ = taufunc(ctx, cb, nodr)
    τ_lr[:, :, 1] = (τ[:, :, 1] .+ τ[:, :, 2]) * 0.5
    τ_lr[:, :, 2] = (τ[:, :, 1] .- τ[:, :, 2]) * 0.5
    E_αm = ephicoef(E_α, nodr)

    for n in 1:nodr
        cin = 4 * ci^(n + 1)
        for p in 1:2
            sp = -monen[p]
            for m in (-n):-1
                pmnp0[n + 1, -m, p, 1] = -cin * τ_lr[n + 1, -m, p] * E_αm[-m]
                pmnp0[n + 1, -m, p, 2] = sp * ci * cin * τ_lr[n + 1, -m, p] * E_αm[-m]
            end

            for m in 0:n
                pmnp0[m, n, p, 1] = -cin * τ_lr[m, n, p] * E_αm[-m]
                pmnp0[m, n, p, 2] = sp * ci * cin * τ_lr[m, n, p] * E_αm[-m]
            end
        end
    end

    return pmnp0
end

"""
    gaussianbeamcoef(ctx, α, β, cbeam, nodr)

Regular VSWF expansion coefficients for a gaussian beam, localized approximation.

cbeam = 1/(k omega)
"""
function gaussianbeamcoef(ctx::ConstantContext, α::Float64, β::Float64, cbeam::Float64, nodr::Int64)
    pmnp0 = planewavecoef(ctx, α, β, nodr)
    for n in 1:nodr
        gbn = exp(-((n + 0.5) * cbeam)^2)
        for p in 1:2
            for k in 1:2
                for m in (-n):-1
                    pmnp0[n + 1, -m, p, k] *= gbn
                end

                for m in 0:n
                    pmnp0[m, n, p, k] *= gbn
                end
            end
        end
    end
    return pmnp0
end

function ephicoef(E_ϕ::ComplexF64, nodr::Int64)
    E_ϕm = OffsetArray(zeros(ComplexF64, 2nodr + 1), (-nodr):nodr)
    E_ϕm[0] = 1.0
    for m in 1:nodr
        E_ϕm[m] = E_ϕ * E_ϕm[m - 1]
        E_ϕm[-m] = conj(E_ϕm[m])
    end
    return E_ϕm
end

"""
    sphereplanewavecoef(ctx, nsphere, neqns, nodr, nodrmax, α, β, rpos, hostsphere, numberfieldexp, rimedium)

Plane wave expansion coefficients at sphere origins. A phase shift is used.
"""
function sphereplanewavecoef(
    ctx::ConstantContext,
    nodr::Array{Int64,1},
    α::Float64,
    β::Float64,
    rpos::Array{Float64,2},
    hostsphere::Array{Int64,1},
    numberfieldexp::Array{Int64,1},
    rimedium::Array{ComplexF64,1},
)
    nsphere = length(nodr)
    nodrmax = maximum(nodr)

    # Valid the input
    @assert length(hostsphere) == nsphere
    @assert length(numberfieldexp) == nsphere
    @assert size(rpos) == (3, nsphere)
    @assert length(rimedium) == 2

    rib = 2 / (1 / rimedium[1] + 1 / rimedium[2])
    pmnp0 = planewavecoef(ctx, α, β, nodrmax)
    cb = cos(β)
    sb = √((1 - cb) * (1 + cb))
    ca = cos(α)
    sa = sin(α)

    neqns = 2 * sum(numberfieldexp[i] * nodr[i] * (nodr[i] + 2) for i in 1:nsphere)
    pmnp = zeros(ComplexF64, 2neqns)

    l = 0
    for i in 1:nsphere
        phasefac = exp(ci * rib * ((ca * rpos[1, i] + sa * rpos[2, i]) * sb + rpos[3, i] * cb))
        for j in 1:numberfieldexp[i]
            for k in 1:2
                for p in 1:2
                    for n in 1:nodr[i]
                        for m in 0:(nodr[i] + 1)
                            l += 1
                            if hostsphere[i] == 0 && j == 1
                                pmnp[l] = phasefac * pmnp0[m, n, p, k]
                            else
                                pmnp[l] = 0
                            end
                        end
                    end
                end
            end
        end
    end

    return pmnp
end

function axialtrancoefrecurrence(
    ctx::ConstantContext,
    itype::Int64,
    r::Float64,
    ri::Array{ComplexF64,1},
    nmax::Int64,
    lmax::Int64,
)
    @assert itype == 1 || itype == 3
    @assert length(ri) == 2

    nlmax = max(nmax, lmax)
    nlmin = min(nmax, lmax)

    nlmax0 = 0
    if nlmax > 0
        nlmax0 = nlmax
        axialtrancoefinit!(ctx, nlmax)
    end

    ndim = atcdim(nmax, lmax)
    ac = zeros(ComplexF64, ndim)
    act = zeros(ComplexF64, nmax, lmax, 2)
    actt = zeros(ComplexF64, 2, 2)

    if iszero(r)
        if itype != 1
            return ac
        end
        iadd0 = 0
        for ma in 0:nlmin
            m1 = max(1, ma)
            for m in (-ma):(2m1):ma
                blockdim = 2(nmax - m1 + 1) * (lmax - m1 + 1)
                iadd1 = iadd0 + blockdim
                for l in m1:nlmax
                    act[l, l, 1] = act[l, l, 2] = 1
                end
                ac[(iadd0 + 1):iadd1] = reshape(act[m1:nmax, m1:lmax, 1:2], (blockdim,))
                iadd0 = iadd1
            end
        end
        return ac
    end

    z = r * ri
    Ξ = OffsetArray(zeros(ComplexF64, nmax + lmax + 1, 2), 0:(nmax + lmax), 1:2)
    for p in 1:2
        Ξ[:, p] = itype == 1 ? cricbessel(nmax + lmax, z[p]) / z[p] : crichankel(nmax + lmax, z[p]) / z[p]
    end

    vcc, fnm1, fn, fnp1 = get_offset_axial_constants(ctx)

    lm1 = lmax - 1
    iadd0 = 0
    for ma in 0:nlmin
        m1 = max(1, ma)
        lp1 = m1 + 1
        for m in (-ma):(2m1):ma
            blockdim = 2(nmax - m1 + 1) * (lmax - m1 + 1)
            iadd1 = iadd0 + blockdim
            n = m1
            for l in m1:lmax
                wmin = abs(n - l)
                wmax = n + l
                ml = l * (l + 1) + m
                for p in 1:2
                    actt[1, p] = actt[2, p] = 0
                    for i in wmin:2:wmax
                        actt[1, p] += vcc[n, ml, i] * Ξ[i, p]
                    end
                    for i in (wmin + 1):2:(wmax - 1)
                        actt[2, p] += vcc[n, ml, i] * Ξ[i, p]
                    end
                    actt[2, p] *= ci
                end
                act[n, l, 1] = actt[1, 1] + actt[2, 1]
                act[n, l, 2] = actt[1, 2] - actt[2, 2]
            end
            l = lmax
            ml = l * (l + 1) + m
            for n in (m1 + 1):nmax
                wmin = abs(n - l)
                wmax = n + l
                for p in 1:2
                    actt[1, p] = actt[2, p] = 0
                    for i in wmin:2:wmax
                        actt[1, p] += vcc[n, ml, i] * Ξ[i, p]
                    end
                    for i in (wmin + 1):2:(wmax - 1)
                        actt[2, p] += vcc[n, ml, i] * Ξ[i, p]
                    end
                    actt[2, p] *= ci
                end
                act[n, l, 1] = actt[1, 1] + actt[2, 1]
                act[n, l, 2] = actt[1, 2] - actt[2, 2]
            end

            if m1 < nlmin
                for n in m1:(nmax - 1)
                    np1 = n + 1
                    nm1 = n - 1
                    for p in 1:2
                        sp = -(-1)^p
                        for i in m1:lm1
                            act[np1, i, p] =
                                -act[n, i + 1, p] * fnp1[m, i] + (fn[m, i] - fn[m, n]) * act[n, i, p] * sp * ci
                        end
                        for i in lp1:lm1
                            act[np1, i, p] += act[n, i - 1, p] * fnm1[m, i]
                        end
                        if n > m1
                            for i in m1:lm1
                                act[np1, i, p] += act[nm1, i, p] * fnm1[m, n]
                            end
                        end
                        for i in m1:lm1
                            act[np1, i, p] /= fnp1[m, n]
                        end
                    end
                end
            end
            ac[(iadd0 + 1):iadd1] = reshape(act[m1:nmax, m1:lmax, 1:2], (blockdim,))
            iadd0 = iadd1
        end
    end

    return ac
end

function axialtrancoefinit!(ctx::ConstantContext, nmax::Int64)
    init!(ctx, nmax)

    ctx.vcc = zeros(nmax, nmax * (nmax + 2), 2nmax + 1)
    ctx.fnm1 = zeros(2nmax + 1, nmax)
    ctx.fn = zeros(2nmax + 1, nmax)
    ctx.fnp1 = zeros(2nmax + 1, nmax)

    _, fnr, monen, _ = get_offset_constants(ctx)
    vcc, fnm1, fn, fnp1 = get_offset_axial_constants(ctx)
    vc1 = OffsetArray(zeros(2nmax + 1), 0:(2nmax))
    vc2 = OffsetArray(zeros(2nmax + 1), 0:(2nmax))
    for n in 1:nmax
        n21 = 2n + 1
        for l in 1:nmax
            c1 = fnr[n21] * fnr[2l + 1]
            ll1 = l * (l + 1)
            vcfunc!(ctx, -1, n, 1, l, vc2)
            wmin = abs(n - l)
            wmax = n + l
            nlmin = min(n, l)
            for m in (-nlmin):nlmin
                ml = ll1 + m
                c2 = -c1 * monen[m]
                vcfunc!(ctx, -m, n, m, l, vc1)
                for w in wmin:wmax
                    inlw = ci^(n - l + w)
                    vcc[n, ml, w] = c2 * vc1[w] * vc2[w] * (real(inlw) + imag(inlw))
                end
            end
        end
    end

    for m in (-nmax):nmax
        for l in max(1, abs(m)):nmax
            lp1 = l + 1
            lm1 = l - 1
            fnm1[m, l] = fnr[lm1] * fnr[lp1] * fnr[l - m] * fnr[l + m] / fnr[lm1 + l] / fnr[lp1 + l] / l
            fn[m, l] = m / l / lp1
            fnp1[m, l] = fnr[l] * fnr[l + 2] * fnr[lp1 - m] * fnr[lp1 + m] / fnr[l + lp1] / fnr[2l + 3] / lp1
        end
    end
end

"""
    tranordertest(ctx, r, ri, lmax, ϵ)

Test to determine convergence of regular VSWF addition theorem for max order lmax and translation distance r w/ refractive index ri.
"""
function tranordertest(ctx::ConstantContext, r::Float64, ri::ComplexF64, lmax::Int64, ϵ::Float64)
    if iszero(r)
        return lmax
    end

    nlim = 200
    init!(ctx, nlim + lmax)
    _, fnr, _, _ = get_offset_constants(ctx)

    z = r * ri
    vc1 = OffsetArray(zeros(nlim + lmax + 1), 0:(nlim + lmax))
    s = 0.0

    for n in 1:nlim
        Ξ = cricbessel(n + lmax, z)
        for l in 0:(n + lmax)
            Ξ[l] = Ξ[l] / z * ci^l
        end
        n21 = 2n + 1
        l = lmax
        c = fnr[n21] * fnr[2l + 1] * ci^(n - l)
        vcfunc!(ctx, -1, n, 1, l, vc1)
        wmin = abs(n - l)
        wmax = n + l
        m = 1
        a = 0.0im
        b = 0.0im
        for w in wmin:wmax
            alnw = vc1[w]^2
            if (n + l + w) & 1 == 0
                a += alnw * Ξ[w]
            else
                b += alnw * Ξ[w]
            end
        end
        a *= c
        b *= c
        s += a * conj(a) + b * conj(b)
        if abs(1.0 - s) < ϵ
            return max(n, lmax)
        end
    end

    return max(nlim, lmax)
end

function atcadd(m::Int64, n::Int64, ntot::Int64)
    return n - ntot + (max(1, m) * (1 + 2ntot - max(1, m))) ÷ 2 + ntot * min(1, m)
end

function atcdim(ntot::Int64, ltot::Int64)
    nmin = min(ntot, ltot)
    nmax = max(ntot, ltot)
    return 2(nmin * (1 - nmin * nmin + 3nmax * (nmin + 2))) ÷ 3
end

function moffset(m::Int64, ntot::Int64, ltot::Int64)
    if m == 0
        return 0
    elseif m < 0
        return 2 * (-((1 + m) * (2 + m) * (3 + 2m + 3ntot)) - 3ltot * (2 + ntot + m * (3 + m + 2ntot))) ÷ 3
    else
        return 2 * (-3ltot * (m - 1)^2 + 6ltot * m * ntot + (m - 1) * (m * (-4 + 2m - 3ntot) + 3(ntot + 1))) ÷ 3
    end
end

"""
    gentrancoef(ctx, itype, xptran, ri, nrow0, nrow1, ncol0, ncol1, iaddrow0, iaddcol0)

Calculates the vwh translation coefficients for a general translation from one origin to another.
"""
function gentrancoef(
    ctx::ConstantContext,
    itype::Int64,
    xptran::Array{Float64,1},
    ri::Array{ComplexF64,1},
    nrow0::Int64,
    nrow1::Int64,
    ncol0::Int64,
    ncol1::Int64,
    iaddrow0::Int64,
    iaddcol0::Int64,
)
    @assert itype == 1 || itype == 3
    @assert length(xptran) == 3
    @assert length(ri) == 2
    @assert 1 <= nrow0 <= nrow1
    @assert 1 <= ncol0 <= ncol1

    ntot = nrow1 + ncol1
    jnc = OffsetArray(zeros(ComplexF64, ntot + 1, 2), 0:ntot, 1:2)
    r, ct, E_ϕ = cartosphere(xptran)
    nblkr0 = (nrow0 - 1) * (nrow0 + 1)
    nblkr1 = nrow1 * (nrow1 + 2)
    nblkc0 = (ncol0 - 1) * (ncol0 + 1)
    nblkc1 = ncol1 * (ncol1 + 2)

    ac = zeros(ComplexF64, 2, nblkr1 - nblkr0 + iaddrow0, nblkc1 - nblkc0 + iaddcol0)

    if iszero(r)
        for n in (nblkr0 + 1):nblkr1
            mn = n - nblkr0 + iaddrow0
            if n > nblkc0 && n <= nblkc1 && itype == 1
                ac[1, mn, n - nblkc0 + iaddcol0] = 1
                ac[2, mn, n - nblkc0 + iaddcol0] = 1
            end
        end
        return ac
    end

    kmax = 0
    ct0 = ct
    drot = rotcoef(ctx, ct0, kmax, ntot)
    E_ϕm = ephicoef(E_ϕ, ntot)
    z = ri * r

    for p in 1:2
        jnc[:, p] = itype == 1 ? cricbessel(ntot, z[p]) / z[p] : crichankel(ntot, z[p]) / z[p]
        for n in 0:ntot
            jnc[n, p] *= ci^n
        end
    end

    _, fnr, monen, _ = get_offset_constants(ctx)

    vc1 = OffsetArray(zeros(ntot + 1), 0:ntot)
    vc2 = OffsetArray(zeros(ntot + 1), 0:ntot)
    atc = zeros(ComplexF64, 2, 2)
    for l in ncol0:ncol1
        ll1 = l * (l + 1)
        for n in nrow0:nrow1
            nn1 = n * (n + 1)
            wmax = n + l
            vcfunc!(ctx, -1, n, 1, l, vc2)
            c = -ci^(n - l) * fnr[2n + 1] * fnr[2l + 1]
            for k in (-l):l
                kl = ll1 + k - nblkc0 + iaddcol0
                for m in (-n):n
                    m1m = monen[m]
                    mn = nn1 + m - nblkr0 + iaddrow0
                    v = k - m
                    vcfunc!(ctx, -m, n, k, l, vc1)
                    a = 0.0
                    b = 0.0
                    wmin = max(abs(v), abs(n - l))
                    for p in 1:2
                        for w in wmax:-1:wmin
                            vw = w * (w + 1) + v
                            if (wmax - w) & 1 == 0
                                a += vc1[w] * vc2[w] * jnc[w, p] * drot[0, vw]
                            else
                                b += vc1[w] * vc2[w] * jnc[w, p] * drot[0, vw]
                            end
                        end
                        atc[p, 1] = a
                        atc[p, 2] = b
                    end
                    ac[1, mn, kl] = (atc[1, 1] + atc[1, 2]) * c * m1m * E_ϕm[v]
                    ac[2, mn, kl] = (atc[2, 1] - atc[2, 2]) * c * m1m * E_ϕm[v]
                end
            end
        end
    end

    return ac
end

function cartosphere(xp::Array{Float64,1})
    @assert length(xp) == 3

    r = xp[1]^2 + xp[2]^2 + xp[3]^2
    if iszero(r)
        return r, 1.0, 1.0 + 0.0im
    end

    r = √r
    ct = xp[3] / r
    E_ϕ = (iszero(xp[1]) && iszero(xp[2])) ? 1.0 + 0.0im : ComplexF64(xp[1], xp[2]) / √(xp[1]^2 + xp[2]^2)

    return r, ct, E_ϕ
end

"""
    eulerrotation(xp)

Euler rotation of a point (xp[1], xp[2], xp[3]).
"""
function eulerrotation(xp::Array{Float64,1}, eulerangf::Array{Float64,1}, dir::Int64)
    @assert length(xp) == 3
    @assert length(eulerangf) == 3

    eulerang = dir == 1 ? eulerangf : -eulerangf[3:-1:1]

    cang = [cos(x) for x in eulerang]
    sang = [sin(x) for x in eulerang]

    mat1 = [[cang[1], -sang[1], 0.0] [sang[1], cang[1], 0.0] [0.0, 0.0, 1.0]]
    mat2 = [[cang[2], 0.0, sang[2]] [0.0, 1.0, 0.0] [-sang[2], 0.0, cang[2]]]
    mat3 = [[cang[3], -sang[3], 0.0] [sang[3], cang[3], 0.0] [0.0, 0.0, 1.0]]

    return mat3 * (mat2 * (mat1 * xp))
end

end # module SpecialFunctions
