function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    gps_estimates = []
    imu_estimates = []
    sqrt_meas_cov = Diagonal([1.0, 1.0])

    fresh_gps_meas = []
    while isready(gps_channel)
        meas = take!(gps_channel)
        push!(fresh_gps_meas, meas)
    end
    fresh_imu_meas = []
    while isready(imu_channel)
        meas = take!(imu_channel)
        push!(fresh_imu_meas, meas)
    end

    if (gps_estimates.isempty())
        gps_estimates.push!(fresh_gps_meas[0])
    end

    if (imu_estimates.isempty())
        imu_estimates.push!(fresh_imu_meas[0])
    end

    @info fresh_gps_meas

    # What are meas_var and cov?
    # Also, all we have for measurements is pos and velocities, how to get quaternions n such?

    mu_hat = rigid_body_dynamics(position, quaternion, velocity, angular_vel, Δt)
    A = jacfx(mu_prev)
    sigma_hat = A * sigma_prev * A' + sqrt_meas_cov # Is this right? taken from measurements module
    
    # z: measurement received
    z = h(xₖ) + sqrt(meas_var)
    C = jac_hx(mu_hat)
    d = h(mu_hat) - C * mu_hat

    # sigma_new: The covariance
    sigma_new = inv(inv(sigma_hat) + C' * inv(meas_var) * C)
    # mu_new: Estimated measurement, different for gps or imu
    mu_new = sigma_new * (inv( sigma_hat) * mu_hat + C' * inv(meas_var) * (z - d))

    localization_state = MyLocalizationType(0,0.0)
    if isready(localization_state_channel)
        take!(localization_state_channel)
    end
    put!(localization_state_channel, localization_state)
end

# Returns a 4x3 matrix the gradient of s;v with respect to r
function gradsv(quaternion, angular_vel, Δt)
    r = angular_vel
    mag = norm(r)

    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * axis
    end

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]
    
    # [Sr; Vr] -> [S; V]
    # Lower right of matrix is Sₙ * I + A
    j4 = [
        sₙ vₙ[1] vₙ[2] vₙ[3];
        vₙ[1] sₙ -vₙ[3] vₙ[2];
        vₙ[2] vₙ[3] sₙ -vₙ[1];
        vₙ[3] -vₙ[2] vₙ[1] sₙ  
        ]

    j1mag = 1/mag * [r[1] r[2] r[3]]
    j2mag = -sin(mag*Δt/2) * Δt/2
    j3mag = sin(mag*Δt/2) * (-r/mag^2) + (r/mag * cos(mag*Δt/2)*Δt/2)

    j2r = [0 0 0]
    #sin(mag*Δt/2)/mag * I
    j3r = [ 
        sin(mag*Δt/2)/mag 0 0;
        0 sin(mag*Δt/2)/mag 0;
        0 0 sin(mag*Δt/2)/mag;
        ]
    grad = j4 * ([j2mag; j3mag] * j1mag + [j2r; j3r]) #This is jac of [s;v] wrt angular v
    return grad
end

function jacfx(position, q, velocity, angular_vel, t)
    r = angular_vel
    mag = norm(r)

    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*t / 2.0)
        vᵣ = sin(mag*t / 2.0) * r/mag
    end

    sₙ = q[1]
    vₙ = q[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    gradrsv = gradsv(q, r, t)

    jac = [
        1 0 0 0 0 0 0 t 0 0 0 0 0;
        0 1 0 0 0 0 0 0 t 0 0 0 0;
        0 0 1 0 0 0 0 0 0 t 0 0 0;
        0 0 0 sᵣ -[1 q[2] q[3]]*vᵣ [q[1] 1 q[3]]*vᵣ [q[1] q[2] 1]*vᵣ 0 0 0 gradrsv[1, 1] gradrsv[1, 2] gradrsv[1, 3]
        0 0 0 vᵣ[1] sᵣ sin(mag*t/2)*r[3]/mag -sin(mag*t/2)*r[2]/mag 0 0 0 gradrsv[2, 1] gradrsv[2, 2] gradrsv[2, 3]
        0 0 0 vᵣ[2] sin(mag*t/2)*r[3]/mag sᵣ -sin(mag*t/2)*r[1]/mag 0 0 0 gradrsv[3, 1] gradrsv[3, 2] gradrsv[3, 3]
        0 0 0 vᵣ[3] sin(mag*t/2)*r[2]/mag -sin(mag*t/2)*r[1]/mag sᵣ 0 0 0 gradrsv[4, 1] gradrsv[4, 2] gradrsv[4, 3]
    ]

    # @info jac
    return jac
end