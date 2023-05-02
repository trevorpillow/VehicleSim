struct MyLocalizationType
    time::Float64
    position::SVector{3,Float64} # Lat and Long
    orientation::SVector{4,Float64}
    linear_vel::SVector{3,Float64}
    angular_vel::SVector{3,Float64}
end

struct MyPerceptionType
    bbox_frame_size::Int
    # bbox_frame_size value explained:
    # -1: no bounding box exists (meaning no neighboring car)
    # 1-50: feel free to speed up or keep the speed
    # 51-80: keep the speed
    # 81-115: slow down
    # >115: stop!!

    # car_to_img_ratio::Float64
    # time::Float64
    # vehicle_id::Int
    # position::SVector{3,Float64} # position of center of vehicle
    # orientation::SVector{4,Float64} # represented as quaternion
    # velocity::SVector{3,Float64}
    # # angular_velocity::SVector{3, Float64} # angular velocity around x,y,z axes
    # size::SVector{3,Float64} # length, width, height of 3d bounding box centered at (position/orientation)
end

function localize(gps_channel, imu_channel, localization_state_channel, gt_channels)
    # Set up algorithm / initialize variables
    x = []
    P = Matrix{Float64}(I, 13, 13)   # 13x13 identity matrix
    Q = Matrix{Float64}(I, 13, 13)   # 13x13 identity matrix
    prevTime = time()

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
       
        newTime = time()
        dt = newTime - prevTime
        imu_meas = zeros(3, 2)
        gps_meas = zeros(3)

        if(!isempty(fresh_imu_meas))
            imu_meas = fresh_imu_meas[end]
        end
        if(!isempty(fresh_gps_meas))
            gps_meas = fresh_gps_meas[end]
        end

        # Initialize state using first GPS measurement if necessary
        if isempty(x) && !isempty(gps_meas)
            x = [gps_meas[1][2:3]; zeros(10, 1)]
        end

        # Predict the state and covariance
        x_pred = f(x, dt)    # Nonlinear state transition function
        F = Jac_x_f(x, dt)   # Jacobian of the state transition function
        P_pred = F * P * F' + Q   # Predicted covariance matrix

        # Predict the measurement and calculate the measurement residual
        if (gps_meas[1] > imu_meas[1])
            # GPS measurement is more recent
            z_pred_gps = h_gps(x_pred)  # Nonlinear measurement function for GPS
            H = Jac_h_gps(x_pred)   # Jacobian of the measurement function for GPS
            residual = gps_meas[2:end] - z_pred_gps    # Measurement residual for GPS
            R = Diagonal([1.0, 1.0]) * 0.01
        else
            # IMU measurement is more recent or they are simultaneous
            z_pred_imu = h_imu(x_pred)  # Nonlinear measurement function for IMU
            H = Jac_h_imu(x_pred)   # Jacobian of the measurement function for IMU
            residual = imu_meas[2:end] - z_pred_imu  # Measurement residual for IMU
            R = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        end

        # Compute the Kalman gain
        S = H * P_pred * H' + R   # Covariance matrix of the measurement residual
        K = P_pred * H' * inv(S)   # Kalman gain

        # Update the state and covariance using the Kalman gain and measurement residual
        x = x_pred + K * residual   # Updated state estimate
        P = (I - K * H) * P_pred   # Updated covariance matrix
        prevTime = newTime # update timeStep
       
        localization_state = MyLocalizationType(x[1:3], x[4:7], x[8:10], x[11:13], newTime)
        if isready(localization_state_channel)
             take!(localization_state_channel)
        end
        put!(localization_state_channel, localization_state)
    end
end

function perception(cam_meas_channel, gt_channel, localization_state_channel, perception_state_channel)
    # set up stuff
    # x0 = [-88, 0, 0.01, 0.01, 13.2, 5.7, 5.3] # EKF depends too heavily on x0
    # image_ratio_width = 640 / 50 # change '40' to something like bbox width or shutdown
    # image_ratio_height = 480 / (2 * bbox[]) # same as above
    # p1_offset = 10 * image_ratio_width
    # p2_offset = 10 * image_ratio_height
    # p1_offset = 10
    # p2_offset = 10
    # prob have to do some sin, cos, velocity stuff?

    # x0 = [x_ego[5] + p1_offset, x_ego[6] + p2_offset, 0.01, 0.01, 13.2, 5.7, 5.3] # [p1 p2 theta vel l w h]
    # println("x0::")
    # println(x0)
    # x0 = [-91.6639496981265, -5.001254676640403, 0.01, 0.01, 13.2, 5.7, 5.3]
    # mus = []
    # sigmas = []

    # gt = fetch(gt_channel)
    # println("gt stuff: id, orientation and position")

    # x_ego_o = gt.orientation
    # x_ego_p = gt.position
    # x_ego = [x_ego_o[1] x_ego_o[2] x_ego_o[3] x_ego_o[4] x_ego_p[1] x_ego_p[2] x_ego_p[3]]
    # println(x_ego)

    # x0 = [x_ego_p[1] + 10, x_ego_p[2] + 10, 0.01, 0.01, 13.2, 5.7, 5.3]
    # delta_t = 0.004
    # current_cam_id = 1
    bbox_frame_size = -1

    while true
        display("in while loop")
        """
            1. Get Camera measurements and ego car's localization
        """
        fresh_cam_meas = []
        while isready(cam_meas_channel)
            # display("cam_meas is ready")
            meas = take!(cam_meas_channel)
            push!(fresh_cam_meas, meas)
        end
        # display("now getting latest_localization_state")
        latest_localization_state = fetch(localization_state_channel)

        # get the cam measurems - five of them 
        # sort them by time
        # so ekf update on them --> might have to add parameters to perception_ekf to take into account of prev ones
        # iterate now
        """
            2. Run ekf
        """
        # NOTE: ONE EKF PER CAM PER VEHICLE SEEN
        # mu_k = zeros(7)
        # closest_bbox = [0.0 0.0 0.0 0.0]
        # frame_size = 0
        # # ny[[-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001]]
        # if length(fresh_cam_meas) > 5
        #     println("fresh_cam_meas length greater than 5")
        #     # deal with the first five camera measurements
        #     for i = 1:5
        #         current_cam_id = fresh_cam_meas[i].camera_id
        #         current_bboxes = fresh_cam_meas[i].bounding_boxes
        #         println("cam id and bbox")
        #         println(current_cam_id)
        #         println(current_bboxes)
        #         # exmaple of current_bboxes: StaticArraysCore.SVector{4, Int64}[[241, 321, 242, 322], [241, 319, 242, 320], [241, 339, 242, 342]]

        #         for j = 1:length(current_bboxes)
        #             one_bbox = current_bboxes[j]
        #             println(one_bbox)
        #             # get the biggest bbox (as it means it's the closest at the moment)
        #             width = abs(one_bbox[2] - one_bbox[4])
        #             height = abs(one_bbox[1] - one_bbox[3])

        #             if width * height > frame_size
        #                 frame_size = width * height
        #                 closest_bbox = current_bboxes[j]
        #             end
        #         end

        #         println("closest_box figured out:")
        #         println(closest_bbox)
        #     end
        # end

        # println("now we try to run ekf")
        # # now get mu and sigma of closest_bbox of one of the cam meas
        # epsilon = 0.00001
        # # only run perception_ekf if we had at least one valid bbox 
        # if (closest_bbox[1] - 0.0 > epsilon) && (closest_bbox[2] - 0.0 > epsilon) && (closest_bbox[3] - 0.0 > epsilon) && (closest_bbox[4] - 0.0 > epsilon)
        #     println("we can!")
        #     mu_k, sigma_k = perception_ekf(x0, closest_bbox, x_ego, delta_t, current_cam_id)
        #     push!(mus, mu_k)
        #     push!(sigmas, sigma_k)
        # end
        # println()
        # println("final mu and sigma")
        # println(mus)
        # println(sigmas)
        # println()
        # # println(mus[length(mus)]) # the last value will be the state of the closest vehicle seen by one of the cams
        # sleep(5)

        """
            Hacky Way
        """
        println("fresh_cam_meas::")
        println(fresh_cam_meas)
        closest_bbox = [0.0 0.0 0.0 0.0]
        frame_size = 0
        # ny[[-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001], [-91.02649421767202, -5.005907915629363, 0.04381970433404755, 0.010203947603373214, 13.199999999999998, 5.7, 5.300000000000001]]
        if length(fresh_cam_meas) > 5
            println("fresh_cam_meas length greater than 5")
            # deal with the first five camera measurements
            for i = 1:5
                current_cam_id = fresh_cam_meas[i].camera_id
                current_bboxes = fresh_cam_meas[i].bounding_boxes
                # println("cam id and bbox")
                # println(current_cam_id)
                # println(current_bboxes)
                # exmaple of current_bboxes: StaticArraysCore.SVector{4, Int64}[[241, 321, 242, 322], [241, 319, 242, 320], [241, 339, 242, 342]]

                for j = 1:length(current_bboxes)
                    one_bbox = current_bboxes[j]
                    println(one_bbox)
                    # get the biggest bbox (as it means it's the closest at the moment)
                    width = abs(one_bbox[2] - one_bbox[4])
                    height = abs(one_bbox[1] - one_bbox[3])

                    if width * height > frame_size
                        frame_size = width * height
                        closest_bbox = current_bboxes[j]
                    end
                end

                println("closest_box figured out:")
                println(closest_bbox)
            end
        end

        epsilon = 0.00001

        if (closest_bbox[1] - 0.0 > epsilon) && (closest_bbox[2] - 0.0 > epsilon) && (closest_bbox[3] - 0.0 > epsilon) && (closest_bbox[4] - 0.0 > epsilon)
            bbox_w = abs(closest_bbox[2] - closest_bbox[4])
            bbox_h = abs(closest_bbox[1] - closest_bbox[3])
            bbox_frame_size = bbox_w * bbox_h

            # car_to_img_ratio_calc = full_frame_size / bbox_frame_size / ratio_constant
        else
            bbox_frame_size = -1
        end

        """
            3. Output the new perception state
        """
        # position = SVector(-91.6639496981265, -5.001254676640403, 2.7)
        # orientation = SVector(0.0, 0.0, 0.0, 0.0)
        # velocity = SVector(0.0, 0.0001, 0.0)
        # size = SVector(13.2, 5.7, 5.3)
        # perception_state = MyPerceptionType(1.06e6, 2, position, orientation, velocity, size)
        perception_state = MyPerceptionType(bbox_frame_size)
        println("perception_state::")
        println(perception_state)
        println()
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
    end
end

function decision_making(localization_state_channel,
    perception_state_channel,
    map_segments,
    target_id,
    socket, gt_channel)

    # do some setup

    # 1. Find current map segment
    gt_meas = take!(gt_channel)
    start_id = find_segment(gt_meas.position[1:2], map_segments)    

    # 2. Find shortest path to target segment
    path, path_index = a_star(map_segments, start_id, target_id)

    # 3. 
    sim(socket, gt_channel, map_segments, path, start_id, target_id)

    # for n in 1:15
    #     while isready(gt_channel)
    #         meas = take!(gt_channel)
    #         if meas.time > gt_meas.time
    #             gt_meas = meas
    #         end
    #     end
    #     #latest_localization_state = fetch(localization_state_channel)
    #     #println(latest_localization_state)

    #     #latest_perception_state = fetch(perception_state_channel)

    #     #gt_meas = take!(gt_channel)
    #     #print(gt_meas.time)
    #     #print("   ")
    #     #println(gt_meas.position[1:2])



    #     # figure out what to do ... setup motion planning problem etc
    #     steering_angle = 0.0
    #     target_vel = 0.0
    #     if n > 5
    #         target_vel = 5.0
    #     end
    #     if n > 10
    #         target_vel = 0.0
    #     end
    #     cmd = VehicleCommand(steering_angle, target_vel, true)
    #     serialize(socket, cmd)
    # end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)

    map_segments = VehicleSim.training_map()

    msg = deserialize(socket) # Visualization info
    @info msg

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{MyLocalizationType}(1)
    perception_state_channel = Channel{MyPerceptionType}(1)

    valid_ids = Condition()
    target_id = 0 # (not a valid segment, will be overwritten by message)
    ego_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    errormonitor(@async while isopen(socket)
        # This while loop reads to the end of the socket stream (makes sure you
        # are looking at the latest messages)
        sleep(0.001)
        local measurement_msg
        received = false
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
                received = true
            else
                break
            end
        end
        !received && continue
        target_id = measurement_msg.target_segment
        ego_id = measurement_msg.vehicle_id
        # notify(valid_ids)
        num_gt = 0

        for meas in measurement_msg.measurements
            # display("meas")
            # display(meas)
            if meas isa GPSMeasurement
                # display("gps meas")
                # display(meas)
                !isfull(gps_channel) && put!(gps_channel, meas)
            elseif meas isa IMUMeasurement
                # display("imu meas")
                # display(meas)
                !isfull(imu_channel) && put!(imu_channel, meas)
            elseif meas isa CameraMeasurement
                # display("CAM MEAS")
                # display(meas)
                !isfull(cam_channel) && put!(cam_channel, meas)
            elseif meas isa GroundTruthMeasurement
                num_gt += 1
                # @info(num_gt)

                !isfull(gt_channel) && put!(gt_channel, meas)
            end
        end
    end)
    
    # wait(valid_ids)
    sleep(2)

    # @async 
    # @async localize(gps_channel, imu_channel, localization_state_channel, gt_channel) #FIXME: Remove gt channel once ready
    # @async perception(cam_channel, localization_state_channel, perception_state_channel)
    t = @async decision_making(localization_state_channel, perception_state_channel, map_segments, target_id, socket, gt_channel)
    errormonitor(t)
    return t

    # @async test_algorithms(gt_channel, localization_state_channel, perception_state_channel, ego_vehicle_id)
end
