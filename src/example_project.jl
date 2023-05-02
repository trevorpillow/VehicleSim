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


function test_algorithms(gt_channel,
    localization_state_channel,
    perception_state_channel,
    ego_vehicle_id)
    estimated_vehicle_states = Dict{Int,Tuple{Float64,Union{SimpleVehicleState,FullVehicleState}}}
    gt_vehicle_states = Dict{Int,GroundTruthMeasurement}

    t = time()
    while true

        while isready(gt_channel)
            meas = take!(gt_channel)
            id = meas.vehicle_id
            if meas.time > gt_vehicle_states[id].time
                gt_vehicle_states[id] = meas
            end
        end

        # Test Localization
        # latest_estimated_ego_state = fetch(localization_state_channel)
        # latest_true_ego_state = gt_vehicle_states[ego_vehicle_id]
        # if latest_estimated_ego_state.last_update < latest_true_ego_state.time - 0.5
        #     @warn "Localization algorithm stale."
        # else
        #     estimated_xyz = latest_estimated_ego_state.position
        #     true_xyz = latest_true_ego_state.position
        #     position_error = norm(estimated_xyz - true_xyz)
        #     t2 = time()
        #     if t2 - t > 5.0
        #         @info "Localization position error: $position_error"
        #         t = t2
        #     end
        # end

        # Test Perception
        latest_perception_state = fetch(perception_state_channel)
        last_perception_update = latest_perception_state.last_update
        vehicles = last_perception_state.x

        for vehicle in vehicles
            xy_position = [vehicle.p1, vehicle.p2]
            closest_id = 0
            closest_dist = Inf
            for (id, gt_vehicle) in gt_vehicle_states
                if id == ego_vehicle_id
                    continue
                else
                    gt_xy_position = gt_vehicle_position[1:2]
                    dist = norm(gt_xy_position - xy_position)
                    if dist < closest_dist
                        closest_id = id
                        closest_dist = dist
                    end
                end
            end
            paired_gt_vehicle = gt_vehicle_states[closest_id]

            # compare estimated to GT

            if last_perception_update < paired_gt_vehicle.time - 0.5
                @info "Perception upate stale"
            else
                # compare estimated to true size
                estimated_size = [vehicle.l, vehicle.w, vehicle.h]
                actual_size = paired_gt_vehicle.size
                @info "Estimated size error: $(norm(actual_size-estimated_size))"
            end
        end
    end
end


function localize(gps_channel, imu_channel, localization_state_channel, gt_channels)

    # @info "Starting localize..."
    lats_longs = [[0.0, 0.0]]
    vels = [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    direction_vectors = [[0.0, 0.0, 0.0]]
    while true
        sleep(0.0001) # Hogs all the cpu without this

        if isready(gps_channel)
            meas = take!(gps_channel)
            push!(lats_longs, [meas.lat, meas.long])

            pos_estimate = mean(lats_longs)
            prev_state = fetch(localization_state_channel)
            delta_pos = [pos_estimate[1] - prev_state.position[1], pos_estimate[2] - prev_state.position[2], 0.0]
            push!(direction_vectors, delta_pos)
        end

        if isready(imu_channel)
            meas = take!(imu_channel)
            push!(vels, [meas.linear_vel, meas.angular_vel])
        end

        if length(lats_longs) > 1
            old_meas = popfirst!(lats_longs)
        end

        if length(vels) > 1
            old_meas = popfirst!(vels)
        end

        pos_estimate = mean(lats_longs)

        if length(direction_vectors) > 50
            old_meas = popfirst!(direction_vectors)
        end

        vels_estimate = mean(vels)
        pos_estimate = [pos_estimate[1], pos_estimate[2], 2.64]

        avg_angle = mean(direction_vectors)
        forward = avg_angle
        # if norm(avg_angle) != 0
        #     forward = avg_angle / norm(avg_angle) # Turning into unit vector makes it NaN
        # end
        up = [0, 0, 1]
        # right = cross(forward, up)
        right = [0.0, 0.0, 0.0]

        gps_offset = Vector([1.0, 3.0, 2.64])
        # directional_offset = gps_offset[1]*forward + gps_offset[2]*right + gps_offset[3]*up
        # if !isnan(directional_offset[1])
        #     pos_estimate = pos_estimate + directional_offset
        # end


        orientation = QuatVec(forward)
        orientation = [orientation[1], orientation[2], orientation[3], orientation[4]]
        localization_state = MyLocalizationType(time(), pos_estimate, orientation, vels_estimate[1], vels_estimate[2])

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

    # valid_ids = Condition()
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

    @async localize(gps_channel, imu_channel, localization_state_channel, gt_channel) #FIXME: Remove gt channel once ready
    @async perception(cam_channel, gt_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map_segments, target_id, socket, gt_channel)

    # @async test_algorithms(gt_channel, localization_state_channel, perception_state_channel, ego_vehicle_id)
end
