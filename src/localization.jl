function rigid_body_dynamics(position, quaternion, velocity, angular_vel, Δt)
    r = angular_vel
    mag = norm(r)

    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * (r / mag)
    end

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    R = Rot_from_quat(quaternion)  

    new_position = position + Δt * R * velocity
    new_quaternion = [s; v]
    new_velocity = velocity
    new_angular_vel = angular_vel
    return [new_position; new_quaternion; new_velocity; new_angular_vel]
end

function f(x, Δt)
    rigid_body_dynamics(x[1:3], x[4:7], x[8:10], x[11:13], Δt)
end

function Jac_x_f(x, Δt)
    J = zeros(13, 13)

    r = x[11:13]
    mag = norm(r)
    if mag < 1e-5
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * (r / mag)
    end

    sₙ = x[4]
    vₙ = x[5:7]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    R = Rot_from_quat([sₙ; vₙ])  
    (J_R_q1, J_R_q2, J_R_q3, J_R_q4) = J_R_q([sₙ; vₙ])
    
    #new_position = position + Δt * R * velocity
    #new_quaternion = [s; v]
    #new_velocity = velocity
    #new_angular_vel = angular_vel
    
    velocity = x[8:10]

    J[1:3, 1:3] = I(3)
    J[1:3, 4] = Δt * J_R_q1*velocity
    J[1:3, 5] = Δt * J_R_q2*velocity
    J[1:3, 6] = Δt * J_R_q3*velocity
    J[1:3, 7] = Δt * J_R_q4*velocity
    J[1:3, 8:10] = Δt * R
    J[4, 4] = sᵣ
    J[4, 5:7] = -vᵣ'
    J[5:7, 4] = vᵣ
    J[5:7, 5:7] = [sᵣ vᵣ[3] -vᵣ[2];
                   -vᵣ[3] sᵣ vᵣ[1];
                   vᵣ[2] -vᵣ[1] sᵣ]

    if mag > 1e-5
        Jsv_srvr = [sₙ -vₙ'
                    vₙ [sₙ -vₙ[3] vₙ[2];
                        vₙ[3] sₙ -vₙ[1];
                        -vₙ[2] vₙ[1] sₙ]]
        Jsrvr_mag = [-sin(mag*Δt / 2.0) * Δt / 2; sin(mag*Δt/2.0) * (-r / mag^2) + cos(mag*Δt/2)*Δt/2 * r/mag]
        Jsrvr_r = [zeros(1,3); sin(mag*Δt / 2) / mag * I(3)]
        Jmag_r = 1/mag * r'

        J[4:7, 11:13] = Jsv_srvr * (Jsrvr_mag*Jmag_r + Jsrvr_r)
    end
    J[8:10, 8:10] = I(3)
    J[11:13, 11:13] = I(3)
    J
end

function h_gps(x)
    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]
    xyz_body = x[1:3] # position
    q_body = x[4:7] # quaternion
    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1]
    yaw = extract_yaw_from_quaternion(q_body)
    meas = [xyz_gps[1:2]; yaw]
end

function Jac_h_gps(x)
    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]
    xyz_body = x[1:3] # position
    q_body = x[4:7] # quaternion
    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1]
    yaw = extract_yaw_from_quaternion(q_body)
    #meas = [xyz_gps[1:2]; yaw]
    J = zeros(3, 13)
    (J_Tbody_xyz, J_Tbody_q) = J_Tbody(x)
    for i = 1:3
        J[1:2,i] = (J_Tbody_xyz[i]*[gps_loc_body; 1])[1:2]
    end
    for i = 1:4
	J[1:2,3+i] = (J_Tbody_q[i]*[gps_loc_body; 1])[1:2]
    end
    w = q_body[1]
    x = q_body[2]
    y = q_body[3]
    z = q_body[4]
    J[3,4] = -(2 * z * (-1 + 2 * (y^2 + z^2)))/(4 * (x * y + w * z)^2 + (1 - 2 * (y^2 + z^2))^2)
    J[3,5] = -(2 * y * (-1 + 2 * (y^2 + z^2)))/(4 * (x * y + w * z)^2 + (1 - 2 * (y^2 + z^2))^2)
    J[3,6] = (2 * (x + 2 * x * y^2 + 4 * w * y * z - 2 * x * z^2))/(1 + 4 * y^4 + 8 * w * x * y * z + 4 * (-1 + w^2) * z^2 + 4 * z^4 + 4 * y^2 * (-1 + x^2 + 2 * z^2))
    J[3,7] = (2 * (w - 2 * w * y^2 + 4 * x * y * z + 2 * w * z^2))/(1 + 4 * y^4 + 8 * w * x * y * z + 4 * (-1 + w^2) * z^2 + 4 * z^4 + 4 * y^2 * (-1 + x^2 + 2 * z^2))
    J
end


function h_imu(x)
    T_body_imu = get_imu_transform()
    T_imu_body = invert_transform(T_body_imu)
    R = T_imu_body[1:3,1:3]
    t = T_imu_body[1:3,end]
    v_body = x[8:10] # linear velocity
    ω_body = x[11:13] # angular velocity
    ω_imu = R * ω_body
    v_imu = R * v_body + t × ω_imu
    meas = [v_imu, ω_imu]
end


function Jac_h_imu(x)
    jac =   [
            0 0 0 0 0 0 0 1 0 -0.02 0 0.7 0;
            0 0 0 0 0 0 0 0 1 0 -0.70028 0 0;
            0 0 0 0 0 0 0 0.02 0 1 0 0.014 0;
            0 0 0 0 0 0 0 0 0 0 1 0 -0.02;
            0 0 0 0 0 0 0 0 0 0 0 1 0;
            0 0 0 0 0 0 0 0 0 0 0.02 0 1
    ]
end

