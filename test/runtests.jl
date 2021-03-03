using Libdl
using MPI
using MSTM
using OffsetArrays
using Test

@testset "MSTM" begin
    MPI.Init()

    mstm_lib = ("CI" => "true") in ENV ? "/home/runner/work/MSTM.jl/MSTM.jl/shared/mstm" : "../shared/mstm"
    mstm = Libdl.dlopen(mstm_lib)

    @testset "Constants" begin
        @testset "init!($notd)" for notd in [5, 10, 20, 50, 100]
            @test begin
                ctx = MSTM.Constants.init(notd)
                bcof_julia, _, _, vwh_coef_julia = MSTM.Constants.get_offset_constants(ctx)

                MSTM.Wrapper.init!(mstm, notd)
                bcof_fortran = MSTM.Wrapper.bcof(mstm, notd)
                vwh_coef_fortran = MSTM.Wrapper.vwh_coef(mstm, notd)

                isapprox(bcof_julia, bcof_fortran) && isapprox(vwh_coef_julia, vwh_coef_fortran)
            end
        end
    end

    @testset "SpecialFunctions" begin
        ϵ = 1e-8
        @testset "ricbessel($n, $ds)" for (n, ds) in [(50, 10.0), (100, 100.0), (150, 200.0)]
            @test begin
                Ψ_julia = MSTM.SpecialFunctions.ricbessel(n, ds, ϵ)
                Ψ_fortran = MSTM.Wrapper.ricbessel(mstm, n, ds, ϵ)
                isapprox(Ψ_julia, Ψ_fortran)
            end
        end

        @testset "richankel($n, $ds)" for (n, ds) in [(50, 10.0), (100, 100.0), (150, 200.0)]
            @test begin
                Ξ_julia = MSTM.SpecialFunctions.richankel(n, ds)
                Ξ_fortran = MSTM.Wrapper.richankel(mstm, n, ds)
                isapprox(Ξ_julia, Ξ_fortran)
            end
        end

        @testset "cricebessel($n, $ds)" for (n, ds) in [
            (10, ComplexF64(0, 0)),
            (20, ComplexF64(10, 10)),
            (500, ComplexF64(0.5, 5.1)),
        ]
            @test begin
                Ψ_julia = MSTM.SpecialFunctions.cricbessel(n, ds)
                Ψ_fortran = MSTM.Wrapper.cricbessel(mstm, n, ds)
                isapprox(Ψ_julia, Ψ_fortran)
            end
        end

        @testset "crichankel($n, $ds)" for (n, ds) in [
            (10, ComplexF64(0, 0)),
            (20, ComplexF64(10, 10)),
            (500, ComplexF64(0.5, 5.1)),
        ]
            @test begin
                Ξ_julia = MSTM.SpecialFunctions.crichankel(n, ds)
                Ξ_fortran = MSTM.Wrapper.crichankel(mstm, n, ds)
                isapprox(Ξ_julia, Ξ_fortran)
            end
        end

        @testset "cspherebessel($n, $z)" for (n, z) in [
            (10, ComplexF64(0, 0)),
            (20, ComplexF64(10, 10)),
            (500, ComplexF64(0.5, 5.1)),
        ]
            @test begin
                csj_julia, csy_julia = MSTM.SpecialFunctions.cspherebessel(n, z)
                csj_fortran, csy_fortran = MSTM.Wrapper.cspherebessel(mstm, n, z)
                isapprox(csj_julia, csj_fortran) && isapprox(csy_julia, csy_fortran)
            end
        end

        @testset "vcfunc($m, $n, $k, $l)" for (m, n, k, l) in [(3, 4, 3, 10), (30, 40, 30, 100), (-4, 10, 4, 5)]
            @test begin
                ctx = MSTM.Constants.init()
                vcn_julia = MSTM.SpecialFunctions.vcfunc(ctx, m, n, k, l)
                vcn_fortran = MSTM.Wrapper.vcfunc(mstm, m, n, k, l)
                isapprox(vcn_julia, vcn_fortran)
            end
        end

        @testset "normalizedlegendre($cbe, $mmax, $nmax)" for (cbe, mmax, nmax) in
                                                              [(0.5, 10, 10), (-0.21, 15, 20), (0.98, 30, 34)]
            @test begin
                ctx = MSTM.Constants.init()
                dc_julia = MSTM.SpecialFunctions.normalizedlegendre(ctx, cbe, mmax, nmax)
                dc_fortran = MSTM.Wrapper.normalizedlegendre(mstm, cbe, mmax, nmax)
                isapprox(dc_julia, dc_fortran)
            end
        end

        @testset "rotcoef($cbe, $kmax, $nmax)" for (cbe, kmax, nmax) in [(0.5, 10, 10), (-0.21, 15, 20), (0.98, 30, 34)]
            @test begin
                ctx = MSTM.Constants.init()
                dc_julia = MSTM.SpecialFunctions.rotcoef(ctx, cbe, kmax, nmax)
                dc_fortran = MSTM.Wrapper.rotcoef(mstm, cbe, kmax, nmax)
                isapprox(dc_julia, dc_fortran)
            end
        end

        @testset "taufunc($cb, $nmax)" for (cb, nmax) in [(0.5, 10), (-0.21, 20), (0.98, 34)]
            @test begin
                ctx = MSTM.Constants.init()
                τ_julia = MSTM.SpecialFunctions.taufunc(ctx, cb, nmax)
                τ_fortran = MSTM.Wrapper.taufunc(mstm, cb, nmax)
                isapprox(τ_julia, τ_fortran)
            end
        end

        @testset "pifunc($cb, $E_ϕ, $nmax, $ndim)" for (cb, E_ϕ, nmax, ndim) in [
            (0.5, 1 + 1.0im, 10, 10),
            (-0.21, 0.2 - 0.3im, 14, 20),
            (0.98, -1.2 + 2.5im, 15, 15),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                π_julia = MSTM.SpecialFunctions.pifunc(ctx, cb, E_ϕ, nmax, ndim)
                π_fortran = MSTM.Wrapper.pifunc(mstm, cb, E_ϕ, nmax, ndim)
                isapprox(π_julia, π_fortran)
            end
        end

        @testset "planewavecoef($α, $β, $nodr)" for (α, β, nodr) in
                                                    [(0.5, 0.6, 10), (-0.21, -0.45, 14), (0.98, -0.45, 20)]
            @test begin
                ctx = MSTM.Constants.init()
                pmnp0_julia = MSTM.SpecialFunctions.planewavecoef(ctx, α, β, nodr)
                pmnp0_fortran = MSTM.Wrapper.planewavecoef(mstm, α, β, nodr)
                isapprox(pmnp0_julia, pmnp0_fortran)
            end
        end

        @testset "gaussianbeamcoef($α, $β, $cbeam, $nodr)" for (α, β, cbeam, nodr) in [
            (0.5, 0.6, 15.0, 10),
            (-0.21, -0.45, 0.5, 14),
            (0.98, -0.45, 100.0, 20),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                pmnp0_julia = MSTM.SpecialFunctions.gaussianbeamcoef(ctx, α, β, cbeam, nodr)
                pmnp0_fortran = MSTM.Wrapper.gaussianbeamcoef(mstm, α, β, cbeam, nodr)
                isapprox(pmnp0_julia, pmnp0_fortran)
            end
        end

        @testset "sphereplanewavecoef" begin
            @test begin
                ctx = MSTM.Constants.init()

                nsphere = 3
                nodr = [3, 3, 3]
                α = 0.5
                β = 0.5
                rpos = [[0.0, 0.0, 0.0] [0.0, 0.0, -0.5] [0.0, 0.0, -1.0]]
                hostsphere = [0, 1, 2]
                numberfieldexp = [3, 3, 3]
                rimedium = [1.1 + 1.3im, 1.0 - 0.5im]

                pmnp_julia = MSTM.SpecialFunctions.sphereplanewavecoef(
                    ctx,
                    nodr,
                    α,
                    β,
                    rpos,
                    hostsphere,
                    numberfieldexp,
                    rimedium,
                )

                pmnp_fortran =
                    MSTM.Wrapper.sphereplanewavecoef(mstm, nodr, α, β, rpos, hostsphere, numberfieldexp, rimedium)

                isapprox(pmnp_julia, pmnp_fortran)
            end
        end

        @testset "axialtrancoefrecurrence($itype, $r, $nmax, $lmax)" for (itype, r, nmax, lmax) in [
            (1, 0.0, 10, 10),
            (1, 1.3, 15, 20),
            (3, 0.0, 4, 5),
            (3, 4.3, 20, 15),
        ]
            @test begin
                ri = [1.0 + 0.5im, 1.0 - 0.5im]
                ctx = MSTM.Constants.init()
                ac_julia = MSTM.SpecialFunctions.axialtrancoefrecurrence(ctx, itype, r, ri, nmax, lmax)
                ac_fortran = MSTM.Wrapper.axialtrancoefrecurrence(mstm, itype, r, ri, nmax, lmax)
                isapprox(ac_julia, ac_fortran)
            end
        end

        @testset "tranordertest($r, $ri, $lmax, $ϵ)" for (r, ri, lmax, ϵ) in [
            (0.01, 1.1 + 0.3im, 10, 1e-4),
            (0.1, 1.1 + 0.03im, 20, 1e-6),
            (0.35, 1.1 + 0.003im, 50, 1e-6),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                nmax_julia = MSTM.SpecialFunctions.tranordertest(ctx, r, ri, lmax, ϵ)
                nmax_fortran = MSTM.Wrapper.tranordertest(mstm, r, ri, lmax, ϵ)
                nmax_julia == nmax_fortran
            end
        end

        @testset "gentrancoef($itype, $xptran, $nrow0, $nrow1, $ncol0, $ncol1, $iaddrow0, $iaddcol0)" for (
            itype,
            xptran,
            nrow0,
            nrow1,
            ncol0,
            ncol1,
            iaddrow0,
            iaddcol0,
        ) in [
            (1, [0.0, 0.0, 0.0], 1, 5, 1, 5, 0, 0),
            (3, [0.0, 0.0, 0.0], 1, 5, 1, 5, 0, 0),
            (1, [1.0, -0.05, 0.04], 1, 5, 1, 5, 0, 0),
            (3, [1.0, -0.05, 0.04], 1, 5, 1, 5, 0, 0),
            (1, [1.0, -0.05, 0.04], 1, 5, 1, 5, 5, 5),
            (3, [1.0, -0.05, 0.04], 1, 5, 1, 5, 2, 3),
        ]
            @test begin
                ri = [1.0 + 0.5im, 1.0 - 0.5im]
                ctx = MSTM.Constants.init()
                ac_julia = MSTM.SpecialFunctions.gentrancoef(
                    ctx,
                    itype,
                    xptran,
                    ri,
                    nrow0,
                    nrow1,
                    ncol0,
                    ncol1,
                    iaddrow0,
                    iaddcol0,
                )
                ac_fortran =
                    MSTM.Wrapper.gentrancoef(mstm, itype, xptran, ri, nrow0, nrow1, ncol0, ncol1, iaddrow0, iaddcol0)
                isapprox(ac_julia, ac_fortran)
            end
        end

        @testset "eulerrotation($xp, $eulerangf, $dir)" for (xp, eulerangf, dir) in [
            ([1.02, 2.03, 3.04], [0.95, 2.33, 1.34], 1),
            ([2.43, -23.2, -19.4], [10.2, 12.3, 11.3], -1),
            ([-0.42, -0.2, -0.001], [-23.2, -23.4, 12.2], 1),
        ]
            @test begin
                xprot_julia = MSTM.SpecialFunctions.eulerrotation(xp, eulerangf, dir)
                xprot_fortran = MSTM.Wrapper.eulerrotation(mstm, xp, eulerangf, dir)
                isapprox(xprot_julia, xprot_fortran)
            end
        end

        @testset "planewavetruncationorder($r, $rimedium, $ϵ)" for (r, rimedium, ϵ) in [
            (0.01, [1.1 + 0.3im, 0.8 - 0.2im], 1e-4),
            (0.1, [1.1 + 0.03im, 1.1 - 0.05im], 1e-6),
            (0.35, [1.1 + 0.003im, 1.2 + 1.1im], 1e-8),
        ]
            @test begin
                nodr_julia = MSTM.SpecialFunctions.planewavetruncationorder(r, rimedium, ϵ)
                nodr_fortran = MSTM.Wrapper.planewavetruncationorder(mstm, r, rimedium, ϵ)
                nodr_julia == nodr_fortran
            end
        end

        @testset "vwhcalc($rpos, $nodr, $itype)" for (rpos, nodr, itype) in [
            ([0.0, 0.0, 0.0], 5, 1),
            ([0.0, 0.0, 0.0], 5, 3),
            ([1.0, -0.05, 0.04], 10, 1),
            ([1.0, -0.05, 0.04], 10, 3),
            ([1.0, -0.05, 0.04], 20, 1),
            ([1.0, -0.05, 0.04], 20, 3),
        ]
            @test begin
                ri = [1.0 + 0.5im, 1.0 - 0.5im]
                ctx = MSTM.Constants.init()
                vwh_julia = MSTM.SpecialFunctions.vwhcalc(ctx, rpos, ri, nodr, itype)
                vwh_fortran = MSTM.Wrapper.vwhcalc(mstm, rpos, ri, nodr, itype)
                isapprox(vwh_julia, vwh_fortran)
            end
        end

        @testset "vwhaxialcalc($rpos, $nodr, $itype)" for (rpos, nodr, itype) in [
            ([0.0, 0.0, 0.0], 5, 1),
            ([0.0, 0.0, 0.0], 5, 3),
            ([1.0, -0.05, 0.04], 10, 1),
            ([1.0, -0.05, 0.04], 10, 3),
            ([1.0, -0.05, 0.04], 20, 1),
            ([1.0, -0.05, 0.04], 20, 3),
        ]
            @test begin
                ri = [1.0 + 0.5im, 1.0 - 0.5im]
                ctx = MSTM.Constants.init()
                vwh_julia = MSTM.SpecialFunctions.vwhaxialcalc(ctx, rpos, ri, nodr, itype)
                vwh_fortran = MSTM.Wrapper.vwhaxialcalc(mstm, rpos, ri, nodr, itype)
                isapprox(vwh_julia, vwh_fortran)
            end
        end

        @testset "twobytwoinverse($mat)" for mat in [
            [1.1+1.0im 2.2-1.5im; 3.3-0.2im -2.5+0.4im],
            [1.1+1.0im 1.2+1.5im; 0.3-0.2im -1.5+0.3im],
        ]
            @test begin
                imat_julia = MSTM.SpecialFunctions.twobytwoinverse(mat)
                imat_fortran = MSTM.Wrapper.twobytwoinverse(mstm, mat)
                isapprox(imat_julia, imat_fortran)
            end
        end
    end

    @testset "Data" begin
        @testset "read MSTM config from .toml file" begin
            config = MSTM.Data.MSTMConfig("fixtures/default.toml")
            @test length(config.spheres) == 2
        end
    end

    @testset "Mie" begin
        ri = OffsetArray(
            [[1.1 + 1.3im, 0.1 + 0.0im] [1.2 + 0.2im, 0.1 + 0.0im] [1.5 - 0.4im, 0.1 + 0.0im] [
                1.5 + 1.5im,
                0.1 + 0.0im,
            ]],
            1:2,
            0:3,
        )

        @testset "mieoa($x, $ri, $nodr0, $qeps, $ri_medium)" for (x, ri, nodr0, qeps, ri_medium) in [
            (0.35, [1.1 + 1.0im, 2.2 - 1.5im], 3, 1e-6, nothing),
            (0.15, [1.1 + 1.0im, 2.2 - 1.5im], 3, 1e-6, [1.0 + 0.1im, 1.0 + 0.1im]),
            (0.25, [1.1 + 1.0im, 2.2 - 1.5im], 4, 0.0, [1.0 + 0.1im, 1.0 + 0.1im]),
            (0.45, [1.1 + 1.0im, 2.2 - 1.5im], 4, -6.0, [1.0 + 0.1im, 1.0 + 0.1im]),
        ]
            @test begin
                qext_julia, qsca_julia, qabs_julia, nodr_julia, _, _, _, _, _, _ =
                    MSTM.Mie.mieoa(x, ri, nodr0, qeps, ri_medium = ri_medium)
                qext_fortran, qsca_fortran, qabs_fortran, nodr_fortran =
                    MSTM.Wrapper.mieoa(mstm, x, ri, nodr0, qeps, ri_medium = ri_medium)
                isapprox(qext_julia, qext_fortran) &&
                    isapprox(qsca_julia, qsca_fortran) &&
                    isapprox(qabs_julia, qabs_fortran) &&
                    nodr_julia == nodr_fortran
            end
        end

        @testset "miecoefcalc($xsp, $hostsphere, $numberfieldexp, $qeps) (No T-Matrix files)" for (
            xsp,
            hostsphere,
            numberfieldexp,
            qeps,
        ) in [
            ([1.2, 0.4, 0.1], [0, 1, 2], [2, 2, 2], 1e-4),
            ([1.2, 0.4, 0.1], [0, 1, 2], [3, 3, 3], 1e-6),
            ([1.2, 0.4, 0.1], [0, 1, 2], [3, 3, 3], 1e-8),
            ([1.2, 0.4, 0.1], [0, 1, 2], [4, 4, 4], 1e-6),
        ]
            @test begin
                params = MSTM.Data.MSTMParameters()
                mie_julia = MSTM.Mie.miecoefcalc(params, xsp, ri, hostsphere, numberfieldexp, qeps)
                mie_fortran = MSTM.Wrapper.miecoefcalc(mstm, xsp, ri, hostsphere, numberfieldexp, qeps)
                isapprox(mie_julia.an, mie_fortran.an) &&
                    isapprox(mie_julia.cn, mie_fortran.cn) &&
                    isapprox(mie_julia.qext, mie_fortran.qext) &&
                    isapprox(mie_julia.qabs, mie_fortran.qabs)
            end
        end

        @testset "onemiecoeffmult($i, $mie_coefficent)" for (i, mie_coefficent) in
                                                            [(1, 'a'), (2, 'c'), (3, 'd'), (2, 'u'), (3, 'v')]
            @test begin
                xsp = [1.2, 0.4, 0.1]
                hostsphere = [0, 1, 2]
                numberfieldexp = [3, 3, 3]
                qeps = 1e-6
                cb = 0.5
                E_ϕ = 1.0 + 1.0im
                params = MSTM.Data.MSTMParameters()
                mie = MSTM.Mie.miecoefcalc(params, xsp, ri, hostsphere, numberfieldexp, qeps)
                nodr = mie.order[i]
                ctx = MSTM.Constants.init()
                MSTM.Wrapper.miecoefcalc(mstm, xsp, ri, hostsphere, numberfieldexp, qeps)
                cx = MSTM.SpecialFunctions.pifunc(ctx, cb, E_ϕ, nodr, nodr)
                cy_julia = MSTM.Mie.onemiecoeffmult(mie, i, cx, mie_coefficent)
                cy_fortran = MSTM.Wrapper.onemiecoeffmult(mstm, i, nodr, cx, mie_coefficent)
                isapprox(cy_julia, cy_fortran)
            end
        end

        @testset "multmiecoeffmult" begin
            xsp = [1.2, 0.4, 0.1]
            hostsphere = [0, 1, 2]
            numberfieldexp = [3, 2, 1]
            qeps = 1e-6
            cb = 0.5
            E_ϕ = 1.0 + 1.0im
            params = MSTM.Data.MSTMParameters()
            mie = MSTM.Mie.miecoefcalc(params, xsp, ri, hostsphere, numberfieldexp, qeps)
            MSTM.Wrapper.miecoefcalc(mstm, xsp, ri, hostsphere, numberfieldexp, qeps)
            for i in 1:4
                @test begin
                    nrhs = i + 3
                    ain = 1.0im * rand(mie.number_eqns * nrhs) + rand(mie.number_eqns * nrhs)
                    aout_julia = MSTM.Mie.multmiecoeffmult(mie, nrhs, i % 2, ain)
                    aout_fortran = MSTM.Wrapper.multmiecoeffmult(mstm, mie.number_eqns, nrhs, i % 2, ain)
                    isapprox(aout_julia, aout_fortran)
                end
            end
        end
    end

    Libdl.dlclose(mstm)
end
