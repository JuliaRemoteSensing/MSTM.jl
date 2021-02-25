module Constants
using OffsetArrays

nmax = 0
bcof = OffsetArray{Float64,2}(undef, 0, 0)
fnr = OffsetArray{Float64,1}(undef, 0)
monen = OffsetArray{Int64,1}(undef, 0)
vwh_coef = OffsetArray{Float64,4}(undef, 0, 0, 0, 0)

function is_inited()
    nmax > 0
end

function reset!()
    nmax = 0
    bcof = OffsetArray{Float64,2}(undef, 0, 0)
    fnr = OffsetArray{Float64,1}(undef, 0)
    monen = OffsetArray{Int64,1}(undef, 0)
    vwh_coef = OffsetArray{Float64,4}(undef, 0, 0, 0, 0)
end

function init!(notd)
    global nmax, bcof, fnr, monen, vwh_coef
    if notd <= nmax
        return
    end
    nmax = notd
    nbc = 6notd + 6
    monen_local = OffsetArray(zeros(Int64, 2notd + 1), 0:2notd)
    bcof_local = OffsetArray(zeros(nbc + 1, nbc + 1), 0:nbc, 0:nbc)
    fnr_local = OffsetArray(zeros(2nbc + 1), 0:2nbc)
    for n in 0:2notd 
        monen_local[n] = n & 1 == 1 ? -1 : 1
    end
    for n in 0:2nbc
        fnr_local[n] = √n
    end
    bcof_local[0, 0] = 1
    for n in 0:nbc - 1
        for l in n + 1:nbc 
            bcof_local[n, l] = fnr_local[n + l] * bcof_local[n, l - 1] / fnr_local[l]
            bcof_local[l, n] = bcof_local[n, l]
        end
        bcof_local[n + 1, n + 1] = fnr_local[2n + 2] * fnr_local[2n + 1] * bcof_local[n, n] / fnr_local[n + 1] / fnr_local[n + 1]
    end

    vwh_coef_local = OffsetArray(zeros(2notd + 1, notd, 3, 3), -notd:notd, 1:notd, -1:1, -1:1)

    for n in 1:notd
        nn1 = n * (n + 1)
        np1 = n + 1
        nm1 = n - 1
        fnorm1 = -0.5 / fnr_local[2n + 1] / fnr_local[n] / fnr_local[n + 1]
        fnorm2 = -0.5 * fnr_local[2n + 1] / fnr_local[n] / fnr_local[n + 1]
        m = -n
        mp1 = m + 1
        mm1 = m - 1
        vwh_coef_local[m,n, 1, 1] = -fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 + mp1]
        vwh_coef_local[m,n, 1,-1] = fnorm1 * np1 * fnr_local[n - m] * fnr_local[nm1 - m]
        vwh_coef_local[m,n,-1, 1] = fnorm1 * n * fnr_local[np1 - m] * fnr_local[np1 - mm1]
        vwh_coef_local[m,n,-1,-1] = 0
        vwh_coef_local[m,n, 0, 1] = fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 - m]
        vwh_coef_local[m,n, 0,-1] = 0
        vwh_coef_local[m,n, 1, 0] = -fnorm2 * fnr_local[n - m] * fnr_local[np1 + m]
        vwh_coef_local[m,n,-1, 0] = -0
        vwh_coef_local[m,n, 0, 0] = -fnorm2 * m
        for m in -n + 1:n - 1
            mp1 = m + 1
            mm1 = m - 1
            vwh_coef_local[m,n, 1, 1] = -fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 + mp1]
            vwh_coef_local[m,n, 1,-1] = fnorm1 * np1 * fnr_local[n - m] * fnr_local[nm1 - m]
            vwh_coef_local[m,n,-1, 1] = fnorm1 * n * fnr_local[np1 - m] * fnr_local[np1 - mm1]
            vwh_coef_local[m,n,-1,-1] = -fnorm1 * np1 * fnr_local[n + m] * fnr_local[nm1 + m]
            vwh_coef_local[m,n, 0, 1] = fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 - m]
            vwh_coef_local[m,n, 0,-1] = fnorm1 * np1 * fnr_local[n + m] * fnr_local[n - m]
            vwh_coef_local[m,n, 1, 0] = -fnorm2 * fnr_local[n - m] * fnr_local[np1 + m]
            vwh_coef_local[m,n,-1, 0] = -fnorm2 * fnr_local[n + m] * fnr_local[np1 - m]
            vwh_coef_local[m,n, 0, 0] = -fnorm2 * m
        end
        m = n
        mp1 = m + 1
        mm1 = m - 1
        vwh_coef_local[m,n, 1, 1] = -fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 + mp1]
        vwh_coef_local[m,n, 1,-1] = 0
        vwh_coef_local[m,n,-1, 1] = fnorm1 * n * fnr_local[np1 - m] * fnr_local[np1 - mm1]
        vwh_coef_local[m,n,-1,-1] = -fnorm1 * np1 * fnr_local[n + m] * fnr_local[nm1 + m]
        vwh_coef_local[m,n, 0, 1] = fnorm1 * n * fnr_local[np1 + m] * fnr_local[np1 - m]
        vwh_coef_local[m,n, 0,-1] = 0
        vwh_coef_local[m,n, 1, 0] = -0
        vwh_coef_local[m,n,-1, 0] = -fnorm2 * fnr_local[n + m] * fnr_local[np1 - m]
        vwh_coef_local[m,n, 0, 0] = -fnorm2 * m
    end

    monen = monen_local
    bcof = bcof_local
    fnr = fnr_local
    vwh_coef = vwh_coef_local
end

export nmax, bcof, fnr, vwh_coef, monen, init!, reset!

end  # module Constants
