module Constants
using OffsetArrays

export ctx, get_offset_constants, get_offset_axial_constants, init!, ConstantContext
mutable struct ConstantContext
    nmax::Int64
    bcof::Array{Float64,2}
    fnr::Array{Float64,1}
    monen::Array{Int64,1}
    vwh_coef::Array{Float64,4}
    vcc::Array{Float64,3}
    fnm1::Array{Float64,2}
    fn::Array{Float64,2}
    fnp1::Array{Float64,2}
end

function get_offset_constants(ctx::ConstantContext)
    notd = ctx.nmax
    nbc = 6notd + 6
    monen = OffsetArray(ctx.monen, (-2notd):(2notd))
    bcof = OffsetArray(ctx.bcof, 0:nbc, 0:nbc)
    fnr = OffsetArray(ctx.fnr, 0:(2nbc))
    vwh_coef = OffsetArray(ctx.vwh_coef, (-notd):notd, 1:notd, -1:1, -1:1)
    return bcof, fnr, monen, vwh_coef
end

function get_offset_axial_constants(ctx::ConstantContext)
    nmax = size(ctx.fn)[2]
    vcc = OffsetArray(ctx.vcc, 1:nmax, 1:(nmax * (nmax + 2)), 0:(2nmax))
    fnm1 = OffsetArray(ctx.fnm1, (-nmax):nmax, 1:nmax)
    fn = OffsetArray(ctx.fn, (-nmax):nmax, 1:nmax)
    fnp1 = OffsetArray(ctx.fnp1, (-nmax):nmax, 1:nmax)
    return vcc, fnm1, fn, fnp1
end

function init()
    ctx = ConstantContext(
        0,
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,1}(undef, 0),
        Array{Int64,1}(undef, 0),
        Array{Float64,4}(undef, 0, 0, 0, 0),
        Array{Float64,3}(undef, 0, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
        Array{Float64,2}(undef, 0, 0),
    )
    return ctx
end

function init(notd)
    ctx = init()
    init!(ctx, notd)
    return ctx
end

function init!(ctx::ConstantContext, notd)
    if notd <= ctx.nmax
        return
    end
    ctx.nmax = notd
    nbc = 6notd + 6
    monen = OffsetArray(zeros(Int64, 4notd + 1), (-2notd):(2notd))
    bcof = OffsetArray(zeros(nbc + 1, nbc + 1), 0:nbc, 0:nbc)
    fnr = OffsetArray(zeros(2nbc + 1), 0:(2nbc))
    monen[-2notd] = 1
    for n in (-2notd + 1):(2notd)
        monen[n] = -monen[n - 1]
    end
    for n in 0:(2nbc)
        fnr[n] = √n
    end
    bcof[0, 0] = 1
    for n in 0:(nbc - 1)
        for l in (n + 1):nbc
            bcof[n, l] = fnr[n + l] * bcof[n, l - 1] / fnr[l]
            bcof[l, n] = bcof[n, l]
        end
        bcof[n + 1, n + 1] = fnr[2n + 2] * fnr[2n + 1] * bcof[n, n] / fnr[n + 1] / fnr[n + 1]
    end

    vwh_coef = OffsetArray(zeros(2notd + 1, notd, 3, 3), (-notd):notd, 1:notd, -1:1, -1:1)

    for n in 1:notd
        nn1 = n * (n + 1)
        np1 = n + 1
        nm1 = n - 1
        fnorm1 = -0.5 / fnr[2n + 1] / fnr[n] / fnr[n + 1]
        fnorm2 = -0.5 * fnr[2n + 1] / fnr[n] / fnr[n + 1]
        m = -n
        mp1 = m + 1
        mm1 = m - 1
        vwh_coef[m, n, 1, 1] = -fnorm1 * n * fnr[np1 + m] * fnr[np1 + mp1]
        vwh_coef[m, n, 1, -1] = fnorm1 * np1 * fnr[n - m] * fnr[nm1 - m]
        vwh_coef[m, n, -1, 1] = fnorm1 * n * fnr[np1 - m] * fnr[np1 - mm1]
        vwh_coef[m, n, -1, -1] = 0
        vwh_coef[m, n, 0, 1] = fnorm1 * n * fnr[np1 + m] * fnr[np1 - m]
        vwh_coef[m, n, 0, -1] = 0
        vwh_coef[m, n, 1, 0] = -fnorm2 * fnr[n - m] * fnr[np1 + m]
        vwh_coef[m, n, -1, 0] = -0
        vwh_coef[m, n, 0, 0] = -fnorm2 * m
        for m in (-n + 1):(n - 1)
            mp1 = m + 1
            mm1 = m - 1
            vwh_coef[m, n, 1, 1] = -fnorm1 * n * fnr[np1 + m] * fnr[np1 + mp1]
            vwh_coef[m, n, 1, -1] = fnorm1 * np1 * fnr[n - m] * fnr[nm1 - m]
            vwh_coef[m, n, -1, 1] = fnorm1 * n * fnr[np1 - m] * fnr[np1 - mm1]
            vwh_coef[m, n, -1, -1] = -fnorm1 * np1 * fnr[n + m] * fnr[nm1 + m]
            vwh_coef[m, n, 0, 1] = fnorm1 * n * fnr[np1 + m] * fnr[np1 - m]
            vwh_coef[m, n, 0, -1] = fnorm1 * np1 * fnr[n + m] * fnr[n - m]
            vwh_coef[m, n, 1, 0] = -fnorm2 * fnr[n - m] * fnr[np1 + m]
            vwh_coef[m, n, -1, 0] = -fnorm2 * fnr[n + m] * fnr[np1 - m]
            vwh_coef[m, n, 0, 0] = -fnorm2 * m
        end
        m = n
        mp1 = m + 1
        mm1 = m - 1
        vwh_coef[m, n, 1, 1] = -fnorm1 * n * fnr[np1 + m] * fnr[np1 + mp1]
        vwh_coef[m, n, 1, -1] = 0
        vwh_coef[m, n, -1, 1] = fnorm1 * n * fnr[np1 - m] * fnr[np1 - mm1]
        vwh_coef[m, n, -1, -1] = -fnorm1 * np1 * fnr[n + m] * fnr[nm1 + m]
        vwh_coef[m, n, 0, 1] = fnorm1 * n * fnr[np1 + m] * fnr[np1 - m]
        vwh_coef[m, n, 0, -1] = 0
        vwh_coef[m, n, 1, 0] = -0
        vwh_coef[m, n, -1, 0] = -fnorm2 * fnr[n + m] * fnr[np1 - m]
        vwh_coef[m, n, 0, 0] = -fnorm2 * m
    end

    ctx.monen = parent(monen)
    ctx.bcof = parent(bcof)
    ctx.fnr = parent(fnr)
    return ctx.vwh_coef = parent(vwh_coef)
end

end  # module Constants
