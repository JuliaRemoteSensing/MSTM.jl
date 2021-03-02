module Mie

using MPI
using OffsetArrays
using Parameters
using ..Constants
using ..Data
using ..SpecialFunctions

export MieData

const ci = 1.0im

@with_kw mutable struct MieData
    number_eqns::Int64 = 0
    max_order::Int64 = 0
    number_spheres::Int64 = 0
    order::Array{Int64,1} = zeros(Int64, 0)
    offset::Array{Int64,1} = zeros(Int64, 0)
    block::Array{Int64,1} = zeros(Int64, 0)
    block_offset::Array{Int64,1} = zeros(Int64, 0)
    number_field_expansions::Array{Int64,1} = zeros(Int64, 0)
    tm_offset::Array{Int64,1} = zeros(Int64, 0)
    is_optically_active::Array{Bool,1} = fill(false, 0)
    tm_on_file::Array{Bool,1} = fill(false, 0)
    qext::Array{Float64,1} = zeros(0)
    qabs::Array{Float64,1} = zeros(0)
    an::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    cn::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    un::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    vn::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    dn::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    an_inv::Array{ComplexF64,1} = zeros(ComplexF64, 0)
    stored_tm::Array{ComplexF64,1} = zeros(ComplexF64, 0)
end

function miecoefcalc(
    run_params::MSTMParameters,
    xsp::Array{Float64,1},
    ri::OffsetArray{ComplexF64,2},
    hostsphere::Array{Int64,1},
    numberfieldexp::Array{Int64,1},
    qeps::Float64;
    tmatrix::Union{TMaxtrix,Nothing} = nothing,
)::MieData
    nsphere = length(xsp)
    @assert size(ri) == (2, nsphere + 1)
    @assert length(hostsphere) == nsphere
    @assert length(numberfieldexp) == nsphere

    mie = MieData()
    mie.max_order = 0
    mie.number_spheres = nsphere
    mie.order = zeros(Int64, nsphere)
    mie.offset = zeros(Int64, nsphere + 1)
    mie.block = zeros(Int64, nsphere)
    mie.block_offset = zeros(Int64, nsphere + 1)
    mie.number_field_expansions = numberfieldexp
    mie.tm_offset = zeros(Int64, nsphere)
    mie.is_optically_active = fill(false, nsphere)
    mie.tm_on_file = fill(false, nsphere)
    mie.qext = zeros(nsphere)
    mie.qabs = zeros(nsphere)

    # TODO: Need to consider communicators other than COMM_WORLD
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    ntermstot = 0
    nblktot = 0
    sizetmstore = 0
    rihost = zeros(ComplexF64, 2)

    # TODO: Implement usage of T-Matrix files
    if !isnothing(tmatrix)
    end

    for i in 1:nsphere
        if !mie.tm_on_file[i]
            rihost = ri[:, hostsphere[i]]
            _, _, _, nodrn, _, _, _, _, _, _ = mieoa(xsp[i], parent(ri[:, i]), 0, qeps, ri_medium = parent(rihost))
            mie.order[i] = nodrn
        end
    end

    # Mie order of host sphere should be no lower than that of the inner sphere
    for i in 1:nsphere
        j = hostsphere[i]
        while j != 0
            mie.order[j] = max(mie.order[j], mie.order[i])
            j = hostsphere[j]
        end
    end

    for i in 1:nsphere
        rihost = ri[:, hostsphere[i]]
        mie.is_optically_active[i] = ri[1, i] != ri[2, i]

        # TODO: Implement usage of T-Matrix files
        if mie.tm_on_file[i]
        else
            qext, qsca, qabs, nodrn, _, _, _, _, _, _ =
                mieoa(xsp[i], parent(ri[:, i]), mie.order[i], 0.0, ri_medium = parent(rihost))
            nterms = 4nodrn
            mie.offset[i] = ntermstot
            ntermstot += nterms
            mie.qext[i] = qext
            mie.qabs[i] = qabs
            mie.block_offset[i] = nblktot
            mie.order[i] = nodrn
            mie.block[i] = nodrn * (nodrn + 2) * 2
        end

        nblktot += mie.block[i] * numberfieldexp[i]
        mie.max_order = max(mie.max_order, mie.order[i])
    end

    mie.offset[nsphere + 1] = ntermstot
    mie.block_offset[nsphere + 1] = nblktot
    mie.number_eqns = nblktot

    mie.an = zeros(ComplexF64, ntermstot)
    mie.cn = zeros(ComplexF64, ntermstot)
    mie.un = zeros(ComplexF64, ntermstot)
    mie.vn = zeros(ComplexF64, ntermstot)
    mie.dn = zeros(ComplexF64, ntermstot)
    mie.an_inv = zeros(ComplexF64, ntermstot)

    # TODO: mieoa() is called 3 times for each sphere, can there be any improvements?
    for i in 1:nsphere
        if !mie.tm_on_file[i]
            rihost = ri[:, hostsphere[i]]
            qext, qsca, qabs, nodrn, anp, cnp, unp, vnp, dnp, anpinv =
                mieoa(xsp[i], parent(ri[:, i]), mie.order[i], 0.0, ri_medium = parent(rihost))
            nterms = 4nodrn
            n1 = mie.offset[i] + 1
            n2 = mie.offset[i] + nterms
            mie.an[n1:n2] = reshape(anp, nterms)
            mie.cn[n1:n2] = reshape(cnp, nterms)
            mie.dn[n1:n2] = reshape(dnp, nterms)
            mie.un[n1:n2] = reshape(unp, nterms)
            mie.vn[n1:n2] = reshape(vnp, nterms)
            mie.an_inv[n1:n2] = reshape(anpinv, nterms)
        end
    end

    return mie
end

"""
    mieoa(x, ri, nodr0, qeps, ri_medium)

Optically active Lorenz/Mie coefficients
"""
function mieoa(
    x::Float64,
    ri::Array{ComplexF64,1},
    nodr0::Int64,
    qeps::Float64;
    ri_medium::Union{Array{ComplexF64,1},Nothing} = nothing,
)
    @assert length(ri) == 2
    @assert isnothing(ri_medium) || length(ri_medium) == 2

    rii = zeros(ComplexF64, 2, 2)
    rii[:, 1] = isnothing(ri_medium) ? [1.0 + 0.0im, 1.0 + 0.0im] : ri_medium
    rii[:, 2] = ri
    ribulk = 2.0 ./ (1.0 ./ rii[1, :] .+ 1.0 ./ rii[2, :])
    xri = x * rii
    nstop = if iszero(qeps)
        nodr0
    elseif qeps > 0.0
        floor(Int, x + 4x^(1 / 3)) + 5
    else
        ceil(Int, -qeps)
    end

    Ψ = OffsetArray(zeros(ComplexF64, nstop + 2, 2, 2), 0:(nstop + 1), 1:2, 1:2)
    Ξ = OffsetArray(zeros(ComplexF64, nstop + 2, 2, 2), 0:(nstop + 1), 1:2, 1:2)
    qext1 = zeros(nstop)
    for i in 1:2
        for p in 1:2
            Ψ[:, p, i] = cricbessel(nstop + 1, xri[p, i])
            Ξ[:, p, i] = crichankel(nstop + 1, xri[p, i])
        end
    end

    qext = 0.0
    qsca = 0.0

    Ψ_p = zeros(ComplexF64, 2, 2)
    Ξ_p = zeros(ComplexF64, 2, 2)
    amat = zeros(ComplexF64, 2, 2)
    bmat = zeros(ComplexF64, 2, 2)
    dmat = zeros(ComplexF64, 2, 2)
    gmat = zeros(ComplexF64, 2, 2)
    umat = zeros(ComplexF64, 2, 2)
    vmat = zeros(ComplexF64, 2, 2)
    anp = zeros(ComplexF64, 2, 2, nstop)
    cnp = zeros(ComplexF64, 2, 2, nstop)
    unp = zeros(ComplexF64, 2, 2, nstop)
    vnp = zeros(ComplexF64, 2, 2, nstop)
    dnp = zeros(ComplexF64, 2, 2, nstop)
    anpinv = zeros(ComplexF64, 2, 2, nstop)

    for n in 1:nstop
        for s in 1:2
            for t in 1:2
                Ψ_p[s, t] = Ψ[n - 1, s, t] - Ψ[n, s, t] / xri[s, t] * n
                Ξ_p[s, t] = Ξ[n - 1, s, t] - Ξ[n, s, t] / xri[s, t] * n
            end
        end

        for s in 1:2
            ss = (-1)^s
            for t in 1:2
                st = (-1)^t
                gmat[s, t] =
                    (ss * ribulk[1] + st * ribulk[2]) / (rii[s, 2] * rii[t, 1]) *
                    (Ψ_p[s, 2] * Ξ[n, t, 1] - ss * st * Ψ[n, s, 2] * Ξ_p[t, 1])
                amat[s, t] =
                    (ss * ribulk[1] + st * ribulk[2]) / (rii[s, 2] * rii[t, 1]) *
                    (Ψ_p[s, 2] * Ψ[n, t, 1] - ss * st * Ψ[n, s, 2] * Ψ_p[t, 1])
                umat[s, t] = (ss * ribulk[2] + st * ribulk[2]) / (rii[s, 2] * rii[t, 2]) * ci
                bmat[s, t] =
                    (ss * ribulk[2] + st * ribulk[1]) / (rii[s, 1] * rii[t, 2]) *
                    (Ξ_p[s, 1] * Ψ[n, t, 2] - ss * st * Ξ[n, s, 1] * Ψ_p[t, 2])
                dmat[s, t] = -(ss * ribulk[1] + st * ribulk[1]) / (rii[s, 1] * rii[t, 1]) * ci
                vmat[s, t] =
                    (ss * ribulk[2] + st * ribulk[1]) / (rii[s, 1] * rii[t, 2]) *
                    (Ξ_p[s, 1] * Ξ[n, t, 2] - ss * st * Ξ[n, s, 1] * Ξ_p[t, 2])
            end
        end

        gmatinv = twobytwoinverse(gmat)
        bmatinv = twobytwoinverse(bmat)

        amat = -gmatinv * amat
        umat = -gmatinv * umat
        dmat = -bmatinv * dmat
        vmat = -bmatinv * vmat

        anp[:, :, n] = amat
        dnp[:, :, n] = dmat
        unp[:, :, n] = umat
        vnp[:, :, n] = vmat
        amatinv = twobytwoinverse(amat)
        cnp[:, :, n] = dmat * amatinv
        amatinv = twobytwoinverse(amat)
        anpinv[:, :, n] = amatinv

        fn1 = 2n + 1
        for p in 1:2
            for q in 1:2
                qsca += fn1 * abs(amat[p, q]) * abs(amat[p, q])
            end
            qext1[n] -= fn1 * real(amat[p, p])
        end
        qext += qext1[n]
    end

    nodr = nstop
    if qeps > 0.0
        qextt = qext
        qext = 0.0
        for n in 1:nstop
            qext += qext1[n]
            err = abs(1.0 - qext / qextt)
            if err < qeps || n == nstop
                nodr = n
                break
            end
        end
    end

    qsca *= 2.0 / x / x
    qext *= 2.0 / x / x
    qabs = qext - qsca

    return qext,
    qsca,
    qabs,
    nodr,
    anp[:, :, 1:nodr],
    cnp[:, :, 1:nodr],
    unp[:, :, 1:nodr],
    vnp[:, :, 1:nodr],
    dnp[:, :, 1:nodr],
    anpinv[:, :, 1:nodr]
end

function onemiecoeffmult(mie::MieData, i::Int64, cx::OffsetArray{ComplexF64,3}, mie_coefficent::Char = 'a')
    @assert length(mie.order) >= i
    nodr = mie.order[i]

    @assert size(cx) == (nodr + 2, nodr, 2)

    nterms = 4nodr
    n1 = mie.offset[i] + 1
    n2 = mie.offset[i] + nterms

    an1 = if mie_coefficent == 'a'
        reshape(mie.an[n1:n2], 2, 2, nodr)
    elseif mie_coefficent == 'c'
        reshape(mie.cn[n1:n2], 2, 2, nodr)
    elseif mie_coefficent == 'd'
        reshape(mie.dn[n1:n2], 2, 2, nodr)
    elseif mie_coefficent == 'u'
        reshape(mie.un[n1:n2], 2, 2, nodr)
    else
        reshape(mie.vn[n1:n2], 2, 2, nodr)
    end

    cy = OffsetArray(zeros(ComplexF64, nodr + 2, nodr, 2), 0:(nodr + 1), 1:nodr, 1:2)

    for n in 1:nodr
        for p in 1:2
            for i in n:-1:1
                cy[n + 1, i, p] = an1[p, 1, n] * cx[n + 1, i, 1] + an1[p, 2, n] * cx[n + 1, i, 2]
            end
            for i in 0:n
                cy[i, n, p] = an1[p, 1, n] * cx[i, n, 1] + an1[p, 2, n] * cx[i, n, 2]
            end
        end
    end

    return cy
end

end # module Mie
