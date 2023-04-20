struct VehicleCommand
    steering_angle::Float64
    velocity::Float64
    controlled::Bool
end

function example_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map_segments = training_map()
    (; chevy_base) = load_mechanism()

    @async while isopen(socket)
        state_msg = deserialize(socket)
    end
   
    shutdown = false
    persist = true
    while isopen(socket)
        position = state_msg.q[5:7]
        @info position
        if norm(position) >= 100
            shutdown = true
            persist = false
        end
        cmd = VehicleCommand(0.0, 2.5, persist, shutdown)
        serialize(socket, cmd) 
    end

end


function get_c()
    ret = ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, true)
    ret == 0 || error("unable to switch to raw mode")
    c = read(stdin, Char)
    ccall(:jl_tty_set_mode, Int32, (Ptr{Cvoid},Int32), stdin.handle, false)
    c
end

function keyboard_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)
    localization_state_channel = Channel{MyLocalizationType}(1)
    put!(localization_state_channel, MyLocalizationType(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])) # fills state channel with dummy value

    @async while isopen(socket)
        sleep(0.001)
        state_msg = deserialize(socket)
        measurements = state_msg.measurements
        num_cam = 0
        num_imu = 0
        num_gps = 0
        num_gt = 0
        for meas in measurements
            if meas isa GroundTruthMeasurement
                !isfull(gt_channel) && put!(gt_channel, meas)
                num_gt += 1
            elseif meas isa CameraMeasurement
                num_cam += 1
            elseif meas isa IMUMeasurement
                !isfull(imu_channel) && put!(imu_channel, meas)
                num_imu += 1
            elseif meas isa GPSMeasurement
                !isfull(gps_channel) && put!(gps_channel, meas)
                num_gps += 1
            end
        end
  #      @info "Measurements received: $num_gt gt; $num_cam cam; $num_imu imu; $num_gps gps"
    end

    target_velocity = 0.0
    steering_angle = 0.0
    controlled = true

    @async localize(gps_channel, imu_channel, localization_state_channel, gt_channel)
    # localize(gps_channel, imu_channel, localization_state_channel, gt_channel)

    @info "Press 'q' at any time to terminate vehicle."
    while controlled && isopen(socket)
        key = get_c()
        if key == 'q'
            # terminate vehicle
            controlled = false
            target_velocity = 0.0
            steering_angle = 0.0
            @info "Terminating Keyboard Client."
        elseif key == 'i'
            # increase target velocity
            target_velocity += v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'k'
            # decrease forward force
            target_velocity -= v_step
            @info "Target velocity: $target_velocity"
        elseif key == 'j'
            # increase steering angle
            steering_angle += s_step
            @info "Target steering angle: $steering_angle"
        elseif key == 'l'
            # decrease steering angle
            steering_angle -= s_step
            @info "Target steering angle: $steering_angle"
        end

        # target_velocity += v_step
        
        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
        
        if isready(localization_state_channel)
            state_est = fetch(localization_state_channel)
            pos_est = state_est.position
            linear_est = state_est.linear_vel
            angular_est = state_est.angular_vel
            orientation_est = state_est.orientation
        end

        gps_offset = Vector([1.0, 3.0, 2.64]) # What is the offset of the GPS relative to the center of the car?
        # offset gps forward 3, down 1, right 1
        # PROBLEM: IMU is always moving "forward" so this calculation is useless
        # forward = Vector(linear_est/norm(linear_est))
        # @info "vel"
        # @info linear_est
        # @info "forward:"
        # @info forward
        # up = [0, 0, 1]
        # right = -cross(forward, up)

        # directional_offset = gps_offset[1]*forward + gps_offset[2]*right + gps_offset[3]*up
        # raw_est = pos_est + [0, 0, 2.64]
        # pos_est = pos_est + directional_offset

        # gt_pos = [0.0, 0.0, 0.0]
        # gt_linear_vel = [0.0, 0.0, 0.0]
        # while isready(gt_channel)
        #     gt_meas = take!(gt_channel)
        #     gt_pos = gt_meas.position
        #     gt_linear_vel = gt_meas.velocity
        #     gt_orientation = gt_meas.orientation
        # end
        # @info "Gt:"
        # @info gt_linear_vel
        # @info "est_orientation:"
        # @info orientation_est
        # @info "offset:"
        # @info gt_pos
        # @info ""
        # @info "raw diff:"
        # @info gt_pos - raw_est
        # @info "offset diff:"
        # @info gt_pos - pos_est
        # @info ""
    end
end

