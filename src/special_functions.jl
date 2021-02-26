module SpecialFunctions
using OffsetArrays
using ..Constants

export ricbessel, richankel, cricbessel, crichankel, cspherebessel, vcfunc, normalizedlegendre

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
    ci = ComplexF64(0, 1)
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
    0.5log10(6.28n) - n * log10(1.36 * x / n)
end

function vcfunc(ctx::ConstantContext, m::Int64, n::Int64, k::Int64, l::Int64)
    init!(ctx, n + l)

    bcof, fnr, _, _ = get_offset_constants(ctx)

    vcn = OffsetArray(zeros(n + l + 1), 0:n + l)
    wmax = n + l
    wmin = max(abs(n - l), abs(m + k))
    vcn[wmax] = bcof[n + m,l + k] * bcof[n - m,l - k] / bcof[n + n,l + l]
    if wmin == wmax
        return vcn
    end
    vcn[wmax - 1] = vcn[wmax] * (l * m - k * n) * fnr[2(l + n) - 1] / fnr[l] / fnr[n] / fnr[n + l + m + k] / fnr[n + l - m - k]
    if wmin == wmax - 1
        return vcn
    end
    mk = m + k
    vcmax = abs(vcn[wmax]) + abs(vcn[wmax - 1])

    # a downwards recurrence is used initially  
    for w in wmax:-1:wmin + 2
        t1::Float64 = 2w * fnr[2w + 1] * fnr[2w - 1] / (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        t3::Float64 = fnr[w - mk - 1] * fnr[w + mk - 1] * fnr[l - n + w - 1] * fnr[n - l + w - 1] * fnr[n + l - w + 2] * fnr[n + l + w] / ((2(w - 1)) * fnr[2w - 3] * fnr[2w - 1])
        vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3
        if (wmax - w) % 2 == 1
            vctest = abs(vcn[w - 2]) + abs(vcn[w - 1])
            vcmax = max(vcmax, vctest)
            rat = vctest / vcmax
            # if/when the coefficients start to decrease in magnitude, an upwards recurrence takes over
            if rat < 0.01 &&  w - 2 > wmin
                wmax = w - 3
                vcfuncuprec(ctx, m, n, k, l, wmax, vcn)
                break
            end
        end
    end

    return vcn
end

"""
    vcfuncuprec(ctx, m, n, k, l, wmax, vcn)

Upwards VC coefficient recurrence, used internally by vcfunc(ctx, m, n, k, l).
"""
function vcfuncuprec(ctx::ConstantContext, m::Int64, n::Int64, k::Int64, l::Int64, wmax::Int64, vcn::OffsetArray{Float64,1})
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
        vc1 = monen[k1 + l1] * bcof[l1 + k1,w - m1 - k1] * bcof[l1 - k1,w + m1 + k1] / bcof[2l1, 2w + 1]
    else
        w = mk
        if m + k >= 0
            vc1 = monen[n + m] * bcof[n - l + w,l - k] * bcof[l - n + w,n - m] / bcof[2w + 1,n + l - w]
        else
            vc1 = monen[l + k] * bcof[n - l + w,l + k] * bcof[l - n + w,n + m] / bcof[2w + 1,n + l - w]
        end
    end
    
    w1 = w
    vcn[w] = vc1
    w = w1 + 1
    mk = m + k
    w2 = min(wmax, n + l)
    if w2 > w1
        t1 = 2 * w * fnr[2w + 1] * fnr[2w - 1] / (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        if w1 == 0
            t2 = .5 * (m - k)
        else
            t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        end
        vcn[w] = t1 * t2 * vcn[w1]
    end
         
    for w in w1 + 2:w2
        t1 = 2w * fnr[2w + 1] * fnr[2w - 1] / (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        t3 = fnr[w - mk - 1] * fnr[w + mk - 1] * fnr[l - n + w - 1] * fnr[n - l + w - 1] * fnr[n + l - w + 2] * fnr[n + l + w] / ((2 * (w - 1)) * fnr[2w - 3] * fnr[2w - 1])
        vcn[w] = t1 * (t2 * vcn[w - 1] - t3 * vcn[w - 2])
    end

    return
end

function normalizedlegendre(ctx::ConstantContext, cbe::Float64, mmax::Int64, nmax::Int64)
    init!(ctx, mmax + nmax)

    bcof, fnr, monen, _ = get_offset_constants(ctx)
    dc = OffsetArray(zeros(2mmax + 1, nmax + 1), -mmax:mmax, 0:nmax)

    # cbe must ∈ [-1, 1] otherwise sbe will be a complex number
    sbe = sqrt((1 + cbe) * (1 - cbe))
    for m in 0:mmax
        dc[m, m] = monen[m] * (0.5 * sbe)^m * bcof[m,m]
        if m == nmax
            break
        end
        dc[m, m + 1] = fnr[2m + 1] * cbe * dc[m, m]
        for n in m + 1:nmax - 1
            dc[m, n + 1] = (-fnr[n - m] * fnr[n + m] * dc[m, n - 1] + (2n + 1) * cbe * dc[m, n]) / (fnr[n + 1 - m] * fnr[n + 1 + m])
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

    dc = OffsetArray(zeros(2kmax + 1, nmax * (nmax + 2) + 1), -kmax:kmax, 0:nmax * (nmax + 2))
    dk0 = OffsetArray(zeros(2nmax + 3), -nmax - 1:nmax + 1)
    dk01 = OffsetArray(zeros(2nmax + 3), -nmax - 1:nmax + 1)

    # cbe must ∈ [-1, 1] otherwise sbe will be a complex number
    sbe = sqrt((1 + cbe) * (1 - cbe))
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

        for k in -n + 1:n - 1
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

            for k in -n:n
                kn = nn1 + k
                dkm1 = dkm0
                dkm0 = dc[m1, kn]
                dkn1 = k == n ? 0.0 : dc[m1, kn + 1]
                dc[m, kn] = (fnr[n + k] * fnr[n - k + 1] * cbe2 * dkm1 - fnr[n - k] * fnr[n + k + 1] * sbe2 * dkn1 - k * sbe * dc[m1, kn]) * fmn
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
    τ = OffsetArray(zeros(nmax + 2, nmax, 2), 0:nmax + 1, 1:nmax, 1:2)
    for n in 1:nmax
        nn1 = n * (n + 1)
        fnm = sqrt((2n + 1) / 2) / 4

        for m in -n:-1
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

    ephim = OffsetArray(zeros(ComplexF64, 2nmax + 1), -nmax:nmax)
    ephim[0] = 1.0

    # ndim should be as large as or larger than nmax
    π_vec = OffsetArray(zeros(ComplexF64, ndim + 2, ndim, 2), 0:ndim + 1, 1:ndim, 1:2)

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
    ci = 1.0im

    τ_lr = OffsetArray(zeros(nodr + 2, nodr, 2), 0:nodr + 1, 1:nodr, 1:2)
    pmnp0 = OffsetArray(zeros(ComplexF64, nodr + 2, nodr, 2, 2), 0:nodr + 1, 1:nodr, 1:2, 1:2)

    cb = cos(β)
    sb = sqrt((1 - cb) * (1 + cb))
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
            for m in -n:-1
                pmnp0[n + 1, -m, p, 1] = -cin * τ_lr[n + 1, -m, p] * E_αm[-m]
                pmnp0[n + 1, -m, p, 2] = sp * ci * cin * τ_lr[n + 1, -m, p] * E_αm[-m]
            end

            for m in 0:n
                pmnp0[m,n,p,1] = -cin * τ_lr[m,n,p] * E_αm[-m]
                pmnp0[m,n,p,2] = sp * ci * cin * τ_lr[m, n, p] * E_αm[-m]
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
                for m in -n:-1
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
    E_ϕm = OffsetArray(zeros(ComplexF64, 2nodr + 1), -nodr:nodr)
    E_ϕm[0] = 1.0
    for m in 1:nodr
        E_ϕm[m] = E_ϕ * E_ϕm[m - 1]
    E_ϕm[-m] = conj(E_ϕm[m])
    end
    return E_ϕm
end

end # module SpecialFunctions
