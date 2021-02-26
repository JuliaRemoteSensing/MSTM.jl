using MSTM
using Test
using Libdl

@testset "MSTM" begin
    mstm_lib = ("CI" => "true") in ENV ? "/home/runner/work/MSTM.jl/MSTM.jl/shared/mstm" : "../shared/mstm"
    mstm = Libdl.dlopen(mstm_lib)

    # @testset "Constants" begin
    #     @testset "init!($notd)" for notd in [5, 10, 20, 50, 100]
    #         @test begin
    #             ctx = MSTM.Constants.init(notd)
    #             bcof_julia, _, _, vwh_coef_julia = MSTM.Constants.get_offset_constants(ctx)

    #             MSTM.Wrapper.init!(mstm, notd)
    #             bcof_fortran = MSTM.Wrapper.bcof(mstm, notd)
    #             vwh_coef_fortran = MSTM.Wrapper.vwh_coef(mstm, notd)

    #             isapprox(bcof_julia, bcof_fortran) && isapprox(vwh_coef_julia, vwh_coef_fortran)
    #         end
    #     end
    # end

    @testset "SpecialFunctions" begin
        ϵ = 1e-8
        @testset "ricbessel($n, $ds)" for (n, ds) in [
            (50, 10.0),
            (100, 100.0),
            (150, 200.0),
        ]
            @test begin
                Ψ_julia = MSTM.SpecialFunctions.ricbessel(n, ds, ϵ)
                Ψ_fortran = MSTM.Wrapper.ricbessel(mstm, n, ds, ϵ)
                isapprox(Ψ_julia, Ψ_fortran)
            end
        end

        @testset "richankel($n, $ds)" for (n, ds) in [
            (50, 10.0),
            (100, 100.0),
            (150, 200.0),
        ]
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

        @testset "vcfunc($m, $n, $k, $l)" for (m, n, k, l) in [
            (3, 4, 3, 10),
            (30, 40, 30, 100),
            (-4, 10, 4, 5),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                vcn_julia = MSTM.SpecialFunctions.vcfunc(ctx, m, n, k, l)
                vcn_fortran = MSTM.Wrapper.vcfunc(mstm, m, n, k, l)
                isapprox(vcn_julia, vcn_fortran)
            end
        end

        @testset "normalizedlegendre($cbe, $mmax, $nmax)" for (cbe, mmax, nmax) in [
            (0.5, 10, 10),
            (-0.21, 15, 20),
            (0.98, 30, 34),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                dc_julia = MSTM.SpecialFunctions.normalizedlegendre(ctx, cbe, mmax, nmax)
                dc_fortran = MSTM.Wrapper.normalizedlegendre(mstm, cbe, mmax, nmax)
                isapprox(dc_julia, dc_fortran)
            end
        end

        @testset "rotcoef($cbe, $kmax, $nmax)" for (cbe, kmax, nmax) in [
            (0.5, 10, 10),
            (-0.21, 15, 20),
            (0.98, 30, 34),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                dc_julia = MSTM.SpecialFunctions.rotcoef(ctx, cbe, kmax, nmax)
                dc_fortran = MSTM.Wrapper.rotcoef(mstm, cbe, kmax, nmax)
                isapprox(dc_julia, dc_fortran)
            end
        end

        @testset "taufunc($cb, $nmax)" for (cb, nmax) in [
            (0.5, 10),
            (-0.21, 20),
            (0.98, 34),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                τ_julia = MSTM.SpecialFunctions.taufunc(ctx, cb, nmax)
                τ_fortran = MSTM.Wrapper.taufunc(mstm, cb, nmax)
                isapprox(τ_julia, τ_fortran)
            end
        end

        @testset "pifunc($cb, $ephi, $nmax, $ndim)" for (cb, ephi, nmax, ndim) in [
            (0.5, 1 + 1.0im, 10, 10),
            (-0.21, 0.2 - 0.3im, 14, 20),
            (0.98, -1.2 + 2.5im, 15, 15),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                π_julia = MSTM.SpecialFunctions.pifunc(ctx, cb, ephi, nmax, ndim)
                π_fortran = MSTM.Wrapper.pifunc(mstm, cb, ephi, nmax, ndim)
                isapprox(π_julia, π_fortran)
            end
        end

        @testset "planewavecoef($α, $β, $nodr)" for (α, β, nodr) in [
            (0.5, 0.6, 10),
            (-0.21, -0.45, 14),
            (0.98, -0.45, 20),
        ]
            @test begin
                ctx = MSTM.Constants.init()
                pmnp0_julia = MSTM.SpecialFunctions.planewavecoef(ctx, α, β, nodr)
                pmnp0_fortran = MSTM.Wrapper.planewavecoef(mstm, α, β, nodr)
                isapprox(pmnp0_julia, pmnp0_fortran)
            end
        end
    end

    Libdl.dlclose(mstm)

end
