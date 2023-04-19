function localize(gps_channel, imu_channel, localization_state_channel)
    # Set up algorithm / initialize variables
    while true
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
        
        # process measurements

        # localization_state = MyLocalizationType(0,0.0)
        # if isready(localization_state_channel)
        #     take!(localization_state_channel)
        # end
        # put!(localization_state_channel, localization_state)
    end 
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

function jacf(position, q, velocity, angular_vel, t)
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
        0 0 0 sᵣ -[1 q[2] q[3]]*vᵣ [q[1] 1 q[3]]*vᵣ [q[1] q[2] 1]*vᵣ 0 0 0 gradrsv[1, 1] gradrsv[1, 2] gradrsv[1, 3];
        0 0 0 vᵣ[1] sᵣ sin(mag*t/2)*r[3]/mag -sin(mag*t/2)*r[2]/mag 0 0 0 gradrsv[2, 1] gradrsv[2, 2] gradrsv[2, 3];
        0 0 0 vᵣ[2] sin(mag*t/2)*r[3]/mag sᵣ -sin(mag*t/2)*r[1]/mag 0 0 0 gradrsv[3, 1] gradrsv[3, 2] gradrsv[3, 3];
        0 0 0 vᵣ[3] sin(mag*t/2)*r[2]/mag -sin(mag*t/2)*r[1]/mag sᵣ 0 0 0 gradrsv[4, 1] gradrsv[4, 2] gradrsv[4, 3]
        0 0 0 0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 0 0 0 1;
    ]

    # @info jac
    return jac
end

function h_gps(x)
    T = get_gps_transform()
    gps_loc_body = T*[zeros(3); 1.0]


    xyz_body = x[1:3] # position
    q_body = x[4:7] # quaternion
    Tbody = get_body_transform(q_body, xyz_body)
    xyz_gps = Tbody * [gps_loc_body; 1]


    gps_meas = xyz_gps[1:2]
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


    imu_meas = [v_imu, ω_imu]
end


function h_imu2(x)
    vx = x[8]
    vy = x[9]
    vz = x[10]
    wx = x[11]
    wy = x[12]
    wz = x[13]
    v_imu_x = vx - 0.02*vz + 0.7*wy
    v_imu_y = vy - 0.70028*wy
    v_imu_z = vz + 0.02*vx +0.014*wy
    w_imu_x = wx - 0.02*wz
    w_imu_y = wy
    w_imu_z = wz + 0.02*wx


    h_imu = [v_imu_x v_imu_y v_imu_z;
            w_imu_x w_imu_y w_imu_z]
end


function h_gps2(x)
    p1 = x[1]
    p2 = x[2]
    p3 = x[3]
    q1 = x[4]
    q2 = x[5]
    q3 = x[6]
    q4 = x[7]
    x_meas = -3(q1^2 + q2^2 - q3^2 - q4^4) + 2(q2*q3 - q1*q4) + 5.2(q1*q3 + q2*q4) + p1
    y_meas = -6(q2*q3 + q1*q4) + (q1^2 - q2^2 + q3^2 - q4^4) + 5.2(q3*q4 - q1*q2) + p2
    #z_meas = -6(q2*q4 - q1*q3) + 2(q1*q2 + q3*q4) + 2.6(q1^2 - q2^2 - q3^2 + q4^4) + p3


    h_gps = [x_meas y_meas]
end


function jac_h_imu(x)
    jac =   [
            0 0 0 0 0 0 0 1 0 -0.02 0 0.7 0;
            0 0 0 0 0 0 0 0 1 0 -0.70028 0 0;
            0 0 0 0 0 0 0 0.02 0 1 0 0.014 0;
            0 0 0 0 0 0 0 0 0 0 1 0 -0.02;
            0 0 0 0 0 0 0 0 0 0 0 1 0;
            0 0 0 0 0 0 0 0 0 0 0.02 0 1
    ]
end


function jac_h_gps(x)
    q1 = x[4]
    q2 = x[5]
    q3 = x[6]
    q4 = x[7]


    #x = [1 0 0 (-6q1 - 2q4 + 5.2q3) (-6q2 + 2q3 + 5.2q4) (6q3 + 2q2 + 5.2q1) (6q4 - 2q1 + 5.2q2)]
    #y = [0 1 0 (-6q4 + 2q1 - 5.2q2) (-6q3 - 2q2 - 5.2q1) (-6q2 + 2q3 + 5.2q4) (-6q1 - 2q4 + 5.2q3)]
    #z = [0 0 1 (6q3 + 2q2 + 5.2q1) (-6q4 + 2q1 - 5.2q2) (6q1 + 2q4 - 5.2q3) (-6q2 + 2q3 + 5.2q4)]


    jac = [
        1 0 0 (-6q1 - 2q4 + 5.2q3) (-6q2 + 2q3 + 5.2q4) (6q3 + 2q2 + 5.2q1) (6q4 - 2q1 + 5.2q2) 0 0 0 0 0 0;
        0 1 0 (-6q4 + 2q1 - 5.2q2) (-6q3 - 2q2 - 5.2q1) (-6q2 + 2q3 + 5.2q4) (-6q1 - 2q4 + 5.2q3) 0 0 0 0 0 0
    ]
end
