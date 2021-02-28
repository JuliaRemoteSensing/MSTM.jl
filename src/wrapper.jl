module Wrapper
using Libdl
using OffsetArrays
using ..SpecialFunctions

# Constants <=> numconstants

function init!(mstm, notd::Int64)
    ccall(Libdl.dlsym(mstm, :__numconstants_MOD_init), Cvoid, (Ref{Int32},), convert(Int32, notd))

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

    return OffsetArray(reshape(vwh_coef, 2notd + 1, notd, 3, 3), (-notd):notd, 1:notd, -1:1, -1:1)
end

# SpecialFunctions <=> specialfuncs

function ricbessel(mstm, n::Int64, ds::Float64, ϵ::Float64)
    Ψ = zeros(n + 1)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_ricbessel),
        Cvoid,
        (Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Int64}, Ptr{Float64}),
        convert(Int32, n),
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
        (Ref{Int32}, Ref{Float64}, Ptr{ComplexF64}),
        convert(Int32, n),
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
        (Ref{Int32}, Ref{ComplexF64}, Ptr{ComplexF64}),
        convert(Int32, n),
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
        (Ref{Int32}, Ref{ComplexF64}, Ptr{ComplexF64}),
        convert(Int32, n),
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
        (Ref{Int32}, Ref{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        convert(Int32, n),
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
        (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Float64}),
        convert(Int32, m),
        convert(Int32, n),
        convert(Int32, k),
        convert(Int32, l),
        vcn,
    )
    return OffsetArray(vcn, 0:(n + l))
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
        dc,
    )
    return OffsetArray(dc, (-mmax):mmax, 0:nmax)
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
        dc,
    )
    return OffsetArray(dc, (-kmax):kmax, 0:(nmax * (nmax + 2)))
end

function taufunc(mstm, cb::Float64, nmax::Int64)
    init!(mstm, nmax + 1)

    τ = zeros(nmax + 2, nmax, 2)
    ccall(Libdl.dlsym(mstm, :__specialfuncs_MOD_taufunc), Cvoid, (Ref{Float64}, Ref{Int64}, Ptr{Float64}), cb, nmax, τ)
    return OffsetArray(τ, 0:(nmax + 1), 1:nmax, 1:2)
end

function pifunc(mstm, cb::Float64, ephi::ComplexF64, nmax::Int64, ndim::Int64)
    init!(mstm, nmax + 1)

    π_vec = zeros(ComplexF64, ndim + 2, ndim, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_pifunc),
        Cvoid,
        (Ref{Float64}, Ref{ComplexF64}, Ref{Int32}, Ref{Int32}, Ptr{ComplexF64}),
        cb,
        ephi,
        convert(Int32, nmax),
        convert(Int32, ndim),
        π_vec,
    )
    return OffsetArray(π_vec, 0:(ndim + 1), 1:ndim, 1:2)
end

function planewavecoef(mstm, α::Float64, β::Float64, nodr::Int64)
    pmnp0 = zeros(ComplexF64, nodr + 2, nodr, 2, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_planewavecoef),
        Cvoid,
        (Ref{Float64}, Ref{Float64}, Ref{Int32}, Ptr{ComplexF64}),
        α,
        β,
        convert(Int32, nodr),
        pmnp0,
    )
    return OffsetArray(pmnp0, 0:(nodr + 1), 1:nodr, 1:2, 1:2)
end

function gaussianbeamcoef(mstm, α::Float64, β::Float64, cbeam::Float64, nodr::Int64)
    pmnp0 = zeros(ComplexF64, nodr + 2, nodr, 2, 2)
    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_gaussianbeamcoef),
        Cvoid,
        (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ptr{ComplexF64}),
        α,
        β,
        cbeam,
        convert(Int32, nodr),
        pmnp0,
    )
    return OffsetArray(pmnp0, 0:(nodr + 1), 1:nodr, 1:2, 1:2)
end

function sphereplanewavecoef(
    mstm,
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

    neqns = 2 * sum(numberfieldexp[i] * nodr[i] * (nodr[i] + 2) for i in 1:nsphere)
    pmnp = zeros(ComplexF64, 2neqns)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_sphereplanewavecoef),
        Cvoid,
        (
            Ref{Int32},
            Ref{Int32},
            Ptr{Int32},
            Ref{Int32},
            Ref{Float64},
            Ref{Float64},
            Ptr{Float64},
            Ptr{Int32},
            Ptr{Int32},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
        ),
        convert(Int32, nsphere),
        convert(Int32, neqns),
        convert(Array{Int32,1}, nodr),
        convert(Int32, nodrmax),
        α,
        β,
        rpos,
        convert(Array{Int32,1}, hostsphere),
        convert(Array{Int32,1}, numberfieldexp),
        rimedium,
        pmnp,
    )

    return pmnp
end

function axialtrancoefrecurrence(mstm, itype::Int64, r::Float64, ri::Array{ComplexF64,1}, nmax::Int64, lmax::Int64)
    @assert itype == 1 || itype == 3
    @assert length(ri) == 2

    init!(mstm, nmax)
    ndim = atcdim(nmax, lmax)
    ac = zeros(ComplexF64, ndim)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_axialtrancoefrecurrence),
        Cvoid,
        (Ref{Int32}, Ref{Float64}, Ptr{ComplexF64}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{ComplexF64}),
        convert(Int32, itype),
        r,
        ri,
        convert(Int32, nmax),
        convert(Int32, lmax),
        convert(Int32, ndim),
        ac,
    )

    return ac
end

function axialtrancoefinit!(mstm, nmax::Int64)
    init!(mstm, nmax)

    ccall(Libdl.dlsym(mstm, :__specialfuncs_MOD_axialtrancoefinit), Cvoid, (Ref{Int32},), convert(Int32, nmax))

    return
end

function tranordertest(mstm, r::Float64, ri::ComplexF64, lmax::Int64, ϵ::Float64)
    nmax = Ref{Int32}(0)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_tranordertest),
        Cvoid,
        (Ref{Float64}, Ref{ComplexF64}, Ref{Int32}, Ref{Float64}, Ref{Int32}),
        r,
        ri,
        convert(Int32, lmax),
        ϵ,
        nmax,
    )

    return convert(Int64, nmax.x)
end

function gentrancoef(
    mstm,
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

    ac = zeros(
        ComplexF64,
        2,
        nrow1 * (nrow1 + 2) - (nrow0 - 1) * (nrow0 - 1) + iaddrow0,
        ncol1 * (ncol1 + 2) - (ncol0 - 1) * (ncol0 + 1) + iaddcol0,
    )

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_gentrancoef),
        Cvoid,
        (
            Ref{Int32},
            Ptr{Float64},
            Ptr{ComplexF64},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ref{Int32},
            Ptr{ComplexF64},
        ),
        convert(Int32, itype),
        xptran,
        ri,
        convert(Int32, nrow0),
        convert(Int32, nrow1),
        convert(Int32, ncol0),
        convert(Int32, ncol1),
        convert(Int32, iaddrow0),
        convert(Int32, iaddcol0),
        ac,
    )

    return ac
end

function eulerrotation(mstm, xp::Array{Float64,1}, eulerangf::Array{Float64,1}, dir::Int64)
    xprot = zeros(3)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_eulerrotation),
        Cvoid,
        (Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ptr{Float64}),
        xp,
        eulerangf,
        convert(Int32, dir),
        xprot,
    )

    return xprot
end

function planewavetruncationorder(mstm, r::Float64, rimedium::Array{ComplexF64, 1}, ϵ::Float64)
    nodr = Ref{Int32}(0)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_planewavetruncationorder),
        Cvoid,
        (Ref{Float64}, Ptr{ComplexF64}, Ref{Float64}, Ref{Int32}),
        r,
        rimedium,
        ϵ,
        nodr,
    )

    return convert(Int64, nodr.x)
end

end # module Wrapper
