module Wrapper
using Libdl
using OffsetArrays

# Constants <=> numconstants

function init!(mstm, notd::Int64)
    ccall(Libdl.dlsym(mstm, :__numconstants_MOD_init), Cvoid, (Ref{Int64},), notd)

    return
end

function bcof(mstm, notd::Int64)
    nbc = 6notd + 6
    bcof_addr_ptr = cglobal(Libdl.dlsym(mstm, :__numconstants_MOD_bcof), UInt64)
    bcof_addr = convert(Ptr{Float64}, unsafe_load(bcof_addr_ptr))
    bcof = unsafe_wrap(Array{Float64,2}, bcof_addr, (nbc + 1, nbc + 1))
    return OffsetArray(bcof, 0:nbc, 0:nbc)
end

function vwh_coef(mstm, notd::Int64)
    vwh_coef_addr_ptr = cglobal(Libdl.dlsym(mstm, :__numconstants_MOD_vwh_coef), UInt64)
    vwh_coef_addr = convert(Ptr{Float64}, unsafe_load(vwh_coef_addr_ptr))
    vwh_coef = unsafe_wrap(Array{Float64}, vwh_coef_addr, ((2notd + 1) * notd * 3 * 3,))

    # FIXME: Exclude outliners as a temporary workaround
    # Need to figure out why there are these strange values
    for idx in eachindex(vwh_coef)
        if abs(vwh_coef[idx]) < 1e-20 || abs(vwh_coef[idx] > 1e20)
            vwh_coef[idx] = 0
        end
    end
    replace!(vwh_coef, NaN => 0)

    return OffsetArray(reshape(vwh_coef, 2notd + 1, notd, 3, 3), (-notd):notd, 1:notd, -1:1, -1:1)
end

# SpecialFunctions <=> specialfuncs

function ricbessel(mstm, n::Int64, ds::Float64, ϵ::Float64)
    Ψ = zeros(n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_ricbessel),
        Cvoid,
        (Ref{Int64}, Ref{Float64}, Ref{Float64}, Ref{Int64}, Ptr{Float64}),
        n,
        ds,
        ϵ,
        0,
        Ψ,
    )
    return OffsetArray(Ψ, 0:n)
end

function richankel(mstm, n::Int64, ds::Float64)
    Ξ = zeros(ComplexF64, n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_richankel),
        Cvoid,
        (Ref{Int64}, Ref{Float64}, Ptr{ComplexF64}),
        n,
        ds,
        Ξ,
    )
    return OffsetArray(Ξ, 0:n)
end

function cricbessel(mstm, n::Int64, ds::ComplexF64)
    Ψ = zeros(ComplexF64, n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_cricbessel),
        Cvoid,
        (Ref{Int64}, Ref{ComplexF64}, Ptr{ComplexF64}),
        n,
        ds,
        Ψ,
    )
    return OffsetArray(Ψ, 0:n)
end

function crichankel(mstm, n::Int64, ds::ComplexF64)
    Ξ = zeros(ComplexF64, n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_crichankel),
        Cvoid,
        (Ref{Int64}, Ref{ComplexF64}, Ptr{ComplexF64}),
        n,
        ds,
        Ξ,
    )
    return OffsetArray(Ξ, 0:n)
end

function cspherebessel(mstm, n::Int64, z::ComplexF64)
    csj = zeros(ComplexF64, n + 1)
    csy = zeros(ComplexF64, n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_cspherebessel),
        Cvoid,
        (Ref{Int64}, Ref{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        n,
        z,
        csj,
        csy,
    )
    return OffsetArray(csj, 0:n), OffsetArray(csy, 0:n)
end

function vcfunc(mstm, m::Int64, n::Int64, k::Int64, l::Int64)
    init!(mstm, n + l)

    vcn = zeros(n + l + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_vcfunc),
        Cvoid,
        (Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ptr{Float64}),
        m,
        n,
        k,
        l,
        vcn
    )
    return OffsetArray(vcn, 0:n + l)
end

function normalizedlegendre(mstm, cbe::Float64, mmax::Int64, nmax::Int64)
    init!(mstm, mmax + nmax)

    dc = zeros(2mmax + 1, nmax + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_normalizedlegendre),
        Cvoid,
        (Ref{Float64}, Ref{Int64}, Ref{Int64}, Ptr{Float64}),
        cbe,
        mmax,
        nmax,
        dc
    )
    return OffsetArray(dc, -mmax:mmax, 0:nmax)
end

function rotcoef(mstm, cbe::Float64, kmax::Int64, nmax::Int64)
    init!(mstm, kmax + nmax)

    dc = zeros(2kmax + 1, nmax * (nmax + 2) + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_rotcoef),
        Cvoid,
        (Ref{Float64}, Ref{Int64}, Ref{Int64}, Ptr{Float64}),
        cbe,
        kmax,
        nmax,
        dc
    )
    return OffsetArray(dc, -kmax:kmax, 0:nmax * (nmax + 2))
end

function taufunc(mstm, cb::Float64, nmax::Int64)
    init!(mstm, nmax + 1)

    τ = zeros(nmax + 2, nmax, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_taufunc),
        Cvoid,
        (Ref{Float64}, Ref{Int64}, Ptr{Float64}),
        cb,
        nmax,
        τ
    )
    return OffsetArray(τ, 0:nmax + 1, 1:nmax, 1:2)
end

function pifunc(mstm, cb::Float64, ephi::ComplexF64, nmax::Int64, ndim::Int64)
    init!(mstm, nmax + 1)

    π_vec = zeros(ComplexF64, ndim + 2, ndim, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_pifunc),
        Cvoid,
        (Ref{Float64}, Ref{ComplexF64}, Ref{Int64}, Ref{Int64}, Ptr{ComplexF64}),
        cb,
        ephi,
        nmax,
        ndim,
        π_vec
    )
    return OffsetArray(π_vec, 0:ndim + 1, 1:ndim, 1:2)
end

function planewavecoef(mstm, α::Float64, β::Float64, nodr::Int64)
    pmnp0 = zeros(ComplexF64, nodr + 2, nodr, 2, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_planewavecoef),
        Cvoid,
        (Ref{Float64}, Ref{Float64}, Ref{Int64},  Ptr{ComplexF64}),
        α,
        β,
        nodr,
        pmnp0
    )
    return OffsetArray(pmnp0, 0:nodr + 1, 1:nodr, 1:2, 1:2)
end

end
