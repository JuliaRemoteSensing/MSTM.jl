module Wrapper
using Libdl
using LinearAlgebra
using OffsetArrays
using ..Mie
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
    ndim = SpecialFunctions.atcdim(nmax, lmax)
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

function planewavetruncationorder(mstm, r::Float64, rimedium::Array{ComplexF64,1}, ϵ::Float64)
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

function vwhcalc(mstm, rpos::Array{Float64,1}, ri::Array{ComplexF64,1}, nodr::Int64, itype::Int64)
    @assert itype == 1 || itype == 3
    @assert length(rpos) == 3
    @assert length(ri) == 2

    vwh = zeros(ComplexF64, 3, 2, nodr * (nodr + 2))

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_vwhcalc),
        Cvoid,
        (Ptr{Float64}, Ptr{ComplexF64}, Ref{Int32}, Ref{Int32}, Ptr{ComplexF64}),
        rpos,
        ri,
        convert(Int32, nodr),
        convert(Int32, itype),
        vwh,
    )

    return vwh
end

function vwhaxialcalc(mstm, rpos::Array{Float64,1}, ri::Array{ComplexF64,1}, nodr::Int64, itype::Int64)
    @assert itype == 1 || itype == 3
    @assert length(rpos) == 3
    @assert length(ri) == 2

    vwh = zeros(ComplexF64, 3, 2, 2, nodr)

    ccall(
        Libdl.dlsym(mstm, :__specialfuncs_MOD_vwhaxialcalc),
        Cvoid,
        (Ptr{Float64}, Ptr{ComplexF64}, Ref{Int32}, Ref{Int32}, Ptr{ComplexF64}),
        rpos,
        ri,
        convert(Int32, nodr),
        convert(Int32, itype),
        vwh,
    )

    return vwh
end

function twobytwoinverse(mstm, mat::Array{ComplexF64,2})
    @assert size(mat) == (2, 2)
    @assert det(mat) != 0

    imat = zeros(ComplexF64, 2, 2)

    ccall(Libdl.dlsym(mstm, :__specialfuncs_MOD_twobytwoinverse), Cvoid, (Ptr{ComplexF64}, Ptr{ComplexF64}), mat, imat)

    return imat
end

# Mie <=> miedata

function miecoefcalc(
    mstm,
    xsp::Array{Float64,1},
    ri::OffsetArray{ComplexF64,2},
    hostsphere::Array{Int64,1},
    numberfieldexp::Array{Int64,1},
    qeps::Float64,
)
    nsphere = length(xsp)
    @assert size(ri) == (2, nsphere + 1)
    @assert length(hostsphere) == nsphere
    @assert length(numberfieldexp) == nsphere

    hostsphere = convert(Array{Int32,1}, hostsphere)
    numberfieldexp = convert(Array{Int32,1}, numberfieldexp)

    tmatrix_file = Ptr{String}()
    ccall(
        Libdl.dlsym(mstm, :__miecoefdata_MOD_miecoefcalc),
        Cvoid,
        (Ref{Int32}, Ptr{Float64}, Ptr{ComplexF64}, Ptr{Int32}, Ptr{Int32}, Ref{Float64}, Ptr{String}),
        convert(Int32, nsphere),
        xsp,
        ri,
        hostsphere,
        numberfieldexp,
        qeps,
        tmatrix_file,
    )

    return getmiedataall(mstm, nsphere)
end

function mieoa(
    mstm,
    x::Float64,
    ri::Array{ComplexF64,1},
    nodr0::Int64,
    qeps::Float64;
    ri_medium::Union{Array{ComplexF64,1},Nothing} = nothing,
)
    @assert length(ri) == 2
    @assert isnothing(ri_medium) || length(ri_medium) == 2

    qext = Ref{Float64}(0.0)
    qsca = Ref{Float64}(0.0)
    qabs = Ref{Float64}(0.0)
    nodr = Ref{Int32}(nodr0)

    # FIXME: present check does not work normally, pass dummy arrays as a workaround
    anp_mie = zeros(ComplexF64, 2, 2, 100)
    dnp_mie = zeros(ComplexF64, 2, 2, 100)
    unp_mie = zeros(ComplexF64, 2, 2, 100)
    vnp_mie = zeros(ComplexF64, 2, 2, 100)
    cnp_mie = zeros(ComplexF64, 2, 2, 100)
    if isnothing(ri_medium)
        ri_medium = [1.0 + 0.0im, 1.0 + 0.0im]
    end
    anp_inv_mie = zeros(ComplexF64, 2, 2, 100)

    ccall(
        Libdl.dlsym(mstm, :__miecoefdata_MOD_mieoa),
        Cvoid,
        (
            Ref{Float64},
            Ptr{ComplexF64},
            Ref{Int32},
            Ref{Float64},
            Ref{Float64},
            Ref{Float64},
            Ref{Float64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
        ),
        x,
        ri,
        nodr,
        qeps,
        qext,
        qsca,
        qabs,
        anp_mie,
        dnp_mie,
        unp_mie,
        vnp_mie,
        cnp_mie,
        ri_medium,
        anp_inv_mie,
    )

    return qext[], qsca[], qabs[], convert(Int64, nodr[])
end

function getmiedataall(mstm, nsphere::Int64)::MieData
    # Do not know termstot at first, so we set an's and cn's sizes to a large number.

    MAXIMUM_TERM = 100_000
    number_eqns = Ref{Int32}(0)
    max_order = Ref{Int32}(0)
    order = zeros(Int32, nsphere)
    offset = zeros(Int32, nsphere + 1)
    block = zeros(Int32, nsphere)
    block_offset = zeros(Int32, nsphere + 1)
    number_field_expansions = zeros(Int32, nsphere)
    is_optically_active = fill(false, nsphere)
    qext = zeros(nsphere)
    qabs = zeros(nsphere)
    an = zeros(ComplexF64, MAXIMUM_TERM)
    cn = zeros(ComplexF64, MAXIMUM_TERM)

    ccall(
        Libdl.dlsym(mstm, :__miecoefdata_MOD_getmiedataallreq),
        Cvoid,
        (
            Ref{Int32},
            Ptr{Int32},
            Ptr{Int32},
            Ptr{Int32},
            Ptr{Int32},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{ComplexF64},
            Ptr{ComplexF64},
            Ref{Int32},
            Ref{Int32},
            Ptr{Bool},
            Ptr{Int32},
        ),
        convert(Int32, nsphere),
        order,
        block,
        offset,
        block_offset,
        qext,
        qabs,
        an,
        cn,
        number_eqns,
        max_order,
        is_optically_active,
        number_field_expansions,
    )

    mie = MieData()
    mie.number_spheres = nsphere
    mie.block = convert(Array{Int64,1}, block)
    mie.offset = convert(Array{Int64,1}, offset)
    mie.block_offset = convert(Array{Int64,1}, block_offset)
    mie.is_optically_active = is_optically_active
    mie.order = convert(Array{Int64,1}, order)
    mie.max_order = convert(Int64, max_order.x)
    mie.number_field_expansions = convert(Array{Int64,1}, number_field_expansions)
    mie.number_eqns = convert(Int32, number_eqns.x)
    mie.qext = qext
    mie.qabs = qabs

    ntermstot = offset[nsphere + 1]
    mie.an = an[1:ntermstot]
    mie.cn = cn[1:ntermstot]
    return mie
end

# # ? <=> spheredata

# function getspheredata(mstm)
#     # Do not know nspheres at first, so we set the arrays' sizes to be a large number.

#     MAXIMUM_NSPHERE = 100_000
#     number_spheres = Ref{Int32}(0)
#     sphere_size_parameters = zeros(MAXIMUM_NSPHERE)
#     sphere_positions = zeros(3, MAXIMUM_NSPHERE)
#     sphere_refractive_indices = OffsetArray(zeros(ComplexF64, 2, MAXIMUM_NSPHERE + 1), 1:2, 0:MAXIMUM_NSPHERE)
#     volume_size_parameter = Ref{Float64}(0.0)
#     host_spheres = zeros(Int32, MAXIMUM_NSPHERE)
#     tmatrix_on_file = fill(false, MAXIMUM_NSPHERE)
#     tmatrix_file = Ptr{String}()

#     ccall(
#         Libdl.dlsym(mstm, :__spheredata_MOD_getspheredata),
#         Cvoid,
#         (Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{ComplexF64}, Ref{Float64}, Ptr{Int32}, Ptr{Bool}, Ptr{String}),
#         number_spheres,
#         Ptr{Float64}(),
#         Ptr{Float64}(),
#         Ptr{ComplexF64}(),
#         volume_size_parameter,
#         Ptr{Int32}(),
#         Ptr{Bool}(),
#         Ptr{String}(),
#         # sphere_size_parameters,
#         # sphere_positions,
#         # sphere_refractive_indices,
#         # volume_size_parameter,
#         # host_spheres,
#         # tmatrix_on_file,
#         # tmatrix_file,
#     )

#     number_spheres = convert(Int64, number_spheres.x)
#     return number_spheres
# end

end # module Wrapper
