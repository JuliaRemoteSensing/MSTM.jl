module Data

using Configurations: Configurations

Configurations.@option "sphere" struct Sphere
    size::Float64
    pos::Array{Float64,1}
    real_ref_index::Union{Float64,Nothing} = nothing
    imag_ref_index::Union{Float64,Nothing} = nothing
    real_chiral_factor::Union{Float64,Nothing} = nothing
    imag_chiral_factor::Union{Float64,Nothing} = nothing
    t_matrix::Union{String,Nothing} = nothing
end

Configurations.@option "run" struct MSTMConfig
    sphere_position_file::Union{String,Nothing} = nothing
    output_file::String = "mstm_default_out.dat"
    run_print_file::Union{String,Nothing} = nothing
    write_sphere_data::Bool = true
    length_scale_factor::Float64 = 1.0
    real_ref_index_scale_factor::Float64 = 1.0
    imag_ref_index_scale_factor::Float64 = 1.0
    real_chiral_factor::Float64 = 0.0
    imag_chiral_factor::Float64 = 0.0
    medium_real_ref_index::Float64 = 1.0
    medium_imag_ref_index::Float64 = 0.0
    medium_real_chiral_factor::Float64 = 0.0
    medium_imag_chiral_factor::Float64 = 0.0
    target_euler_angles_deg::Array{Float64,1} = [0.0, 0.0, 0.0]
    mie_epsilon::Float64 = 1e-6
    translation_epsilon::Float64 = 1e-6
    solution_epsilon::Float64 = 1e-8
    max_number_iterations::Int64 = 5000
    store_translation_matrix::Bool = false
    sm_number_processors::Int64 = 10
    near_field_distance::Float64 = 1e8
    iterations_per_correction::Int64 = 20
    min_scattering_angle_deg::Float64 = 0.0
    max_scattering_angle_deg::Float64 = 180.0
    min_scattering_plane_angle_deg::Float64 = 0.0
    max_scattering_plane_angle_deg::Float64 = 0.0
    delta_scattering_angle_deg::Float64 = 1.0
    frame::String = "target"
    normalize_scattering_matrix::Bool = true
    gaussian_beam_constant::Float64 = 0.0
    gaussian_beam_focal_point::Array{Float64,1} = [0.0, 0.0, 0.0]
    orientation::String = "fixed"
    incident_azimuth_angle_deg::Float64 = 0.0
    incident_polar_angle_deg::Float64 = 0.0
    calculate_scattering_coefficients::Bool = true
    scattering_coefficient_file::String = ""
    track_iterations::Bool = true
    calculate_near_field::Bool = true
    near_field_plane_coord::Int64 = 1
    near_field_plane_vertices::Array{Float64,1} = [10.0, 10.0]
    spacial_step_size::Float64 = 0.5
    polarization_angle_deg::Float64 = 0.0
    near_field_output_file::String = "nf_default_out.dat"
    near_field_output_data::Int64 = 2
    plane_wave_epsilon::Float64 = 1e-4
    calculate_t_matrix::Bool = true
    t_matrix_file::String = "tm_default.dat"
    t_matrix_convergence_epsilon::Float64 = 1e-7
    spheres::Array{Sphere}
end

function from_file(filename::String)
    return Configurations.from_toml(MSTMConfig, filename)
end

end # module Data
