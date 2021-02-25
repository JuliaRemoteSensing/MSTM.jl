module SpecialFunctions
using OffsetArrays
using MSTM.Constants

export ricbessel

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

function vcfunc(m::Int64, n::Int64, k::Int64, l::Int64)
    init!(n + l)

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
        t1 = 2w * fnr[2w + 1] * fnr[2w - 1] / (fnr[w + mk] * fnr[w - mk] * fnr[n - l + w] * fnr[l - n + w] * fnr[n + l - w + 1] * fnr[n + l + w + 1])
        t2 = ((m - k) * w * (w - 1) - mk * n * (n + 1) + mk * l * (l + 1)) / (2w * (w - 1))
        t3 = fnr[w - mk - 1] * fnr[w + mk - 1] * fnr[l - n + w - 1] * fnr[n - l + w - 1] * fnr[n + l - w + 2] * fnr[n + l + w] / ((2(w - 1)) * fnr[2w - 3] * fnr[2w - 1])
        vcn[w - 2] = (t2 * vcn[w - 1] - vcn[w] / t1) / t3
        if (wmax - w) % 2 == 1
            vctest = abs(vcn[w - 2]) + abs(vcn[w - 1])
            vcmax = max(vcmax, vctest)
            rat = vctest / vcmax
            # if/when the coefficients start to decrease in magnitude, an upwards recurrence takes over
            if rat < 0.01 &&  w - 2 > wmin
                wmax = w - 3
                vcfuncuprec(m, n, k, l, wmax, vcn)
                break
            end
        end
    end

    return vcn
end

function vcfuncuprec(m::Int64, n::Int64, k::Int64, l::Int64, wmax::Int64, vcn::OffsetArray{Float64,1})
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
        vc1 = (-1)^(k1 + l1) * bcof[l1 + k1,w - m1 - k1] * bcof[l1 - k1,w + m1 + k1] / bcof[2l1, 2w + 1]
    else
        w = mk
        if m + k >= 0
            vc1 = (-1)^(n + m) * bcof[n - l + w,l - k] * bcof[l - n + w,n - m] / bcof[2w + 1,n + l - w]
        else
            vc1 = (-1)^(l + k) * bcof[n - l + w,l + k] * bcof[l - n + w,n + m] / bcof[2w + 1,n + l - w]
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

end # module SpecialFunctions
