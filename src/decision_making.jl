function sim(socket, gt_channel, map_segments, path, start_id, target_id;
        timestep = 0.1,
        trajectory_length = 10,
        R = Diagonal([0.05, 0.7]))
    
    # Setup callbacks appropriately
    callbacks = callback_generator(trajectory_length, timestep, R)

    # Get GT measurement
    wait(gt_channel)
    gt_meas = fetch(gt_channel)

    # if isready(gt_channel)
    #     gt_meas = fetch(gt_channel)
    # else
    #     wait(gt_channel)
    #     gt_meas = fetch(gt_channel)
    # end

    # Calculate direction from quaternion
    w = gt_meas.orientation[1]
    x = gt_meas.orientation[2]
    y = gt_meas.orientation[3]
    z = gt_meas.orientation[4]
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)

    # Initial state
    ego_state = [gt_meas.position[1:2]; atan(t3, t4); gt_meas.velocity[1]; gt_meas.angular_velocity[3]]

    # Keep track of current segment, next segment, and next turn segment
    segment_id = start_id
    next_turn_id = start_id
    child_id = path[segment_id]

    h_turn = halfspace(map_segments[next_turn_id].lane_boundaries[1].pt_a, map_segments[next_turn_id].lane_boundaries[2].pt_a)
    at = [h_turn.a[1];h_turn.a[2]]
    bt = h_turn.b

    segs = [;]
    for k in 1:trajectory_length
        push!(segs, start_id)
    end
    
    
    # stale = 0

    # Array of points on path for plotting
    points = [;]
    push!(points, ego_state[1:2])

    for t in 1:1000
        # Reading measurements commented out for debugging

        # sleep(0.01)
        
        # # if stale >= 5
        # #     wait(gt_channel)
        # # end
        # received = false
        # if isready(gt_channel)
        #     # stale = 0
        #     while isready(gt_channel)
        #         meas = take!(gt_channel)
        #         if meas.time > gt_meas.time
        #             # @info("GOT MEAS")
        #             # @info(meas.time - gt_meas.time)
        #             gt_meas = meas
        #             received = true
        #         end
        #     end
        # # else
        # #     stale += 1
        # end

        # # wait(gt_channel)
        # # while isready(gt_channel)
        # #     meas = take!(gt_channel)
        # #     if meas.time > gt_meas.time
        # #         gt_meas = meas
        # #         # @info("GOT MEAS")
        # #     end
        # # end

        
        # if received
        #     ego_state = [gt_meas.position[1:2]; 2 * acos(gt_meas.orientation[1]); gt_meas.velocity[1]; gt_meas.angular_velocity[3]]
        # end

        # Update current, next, and next turn segments
        new_id = find_segment(ego_state[1:2], map_segments, keys(path))
        if new_id == child_id
            segment_id = new_id
            child_id = path[segment_id]
        end

        segment = map_segments[segment_id]

        if next_turn_id == segment_id
            next_turn_id = path[next_turn_id]
            if next_turn_id == 0
                break
            end

            while next_turn_id != target_id && isapprox(map_segments[next_turn_id].lane_boundaries[1].curvature, 0.0, atol=1e-6)
                next_turn_id = path[next_turn_id]
            end
            h_turn = halfspace(map_segments[next_turn_id].lane_boundaries[1].pt_a, map_segments[next_turn_id].lane_boundaries[2].pt_a)
            at = [h_turn.a[1];h_turn.a[2]]
            bt = h_turn.b
        end

        # There has to be a better way to do this...
        # but using arrays like this sorta works, so I haven't changed it

        s = [;]  # 1 if straight, 0 if not
        t = [;]  # 1 if turn, 0 if not
        a1 = [;] # left boundary vectors
        b1 = [;] # left boundary values
        a2 = [;] # right boundary vectors
        b2 = [;] # right boundary values
        c = [;]  # circle centers
        ri = [;] # inside radii
        ro = [;] # outside radii

        # Fill arrays for each segment step
        for k in 1:trajectory_length
            segment = map_segments[segs[k]]

            if isapprox(segment.lane_boundaries[1].curvature, 0.0, atol=1e-6)
                straight = 1.0
                turn = 0.0
            else
                straight = 0.0
                turn = 1.0
            end

            h_left = halfspace(segment.lane_boundaries[1].pt_a, segment.lane_boundaries[1].pt_b)
            h_right = halfspace(segment.lane_boundaries[2].pt_a, segment.lane_boundaries[2].pt_b)
            center = center_of_curve(segment)
            r_inside, r_outside = radii(segment)

            push!(s, straight)
            push!(t, turn)
            push!(a1, h_left.a[1])
            push!(a1, h_left.a[2])
            push!(b1, h_left.b)
            push!(a2, h_right.a[1])
            push!(a2, h_right.a[2])
            push!(b2, h_right.b)
            push!(c, center[1])
            push!(c, center[2])
            push!(ri, r_inside)
            push!(ro, r_outside)
        end

        # Generate trajectory
        trajectory = generate_trajectory(ego_state, s, t, a1, b1, a2, b2, c, ri, ro, at, bt, callbacks, trajectory_length)

        # @info(trajectory.states)
        # @info(trajectory.controls)

        velocity, steering_angle = update_controls(ego_state, trajectory.controls[1], timestep)

        # Commented out to debug trajectory optimization
        # cmd = VehicleCommand(steering_angle, velocity, true)
        # serialize(socket, cmd)

        ego_state = trajectory.states[1]
        push!(points, ego_state[1:2])

        # Update segments for all trajectory steps
        for k in 1:trajectory_length
            # @info(trajectory.controls[k][1:2])
            # @info(trajectory.states[k])
            # @info(find_segment(trajectory.states[k][1:2], map_segments, keys(path)))
            if find_segment(trajectory.states[k][1:2], map_segments, keys(path)) == child_id
                segs[k] = child_id
            else
                segs[k] = segment_id
            end
        end


        # if find_segment(ego_state[1:2], map_segments, keys(path)) == 0
        #     @info(find_segment(ego_state[1:2], map_segments, keys(path)))
        #     @info(trajectory.states)
        #     @info(trajectory.controls)
        #     # break
        # end
        
    end

    # @info(points)
    plot_path(map_segments, points)
end

function generate_trajectory(X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, callbacks, trajectory_length)

    wrapper_f = function(Z)
        callbacks.full_cost_fn(X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z)
    end
    wrapper_grad_f = function(Z, grad)
        callbacks.full_cost_grad_fn(grad, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z)
    end
    wrapper_con = function(Z, con)
        callbacks.full_constraint_fn(con, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z)
    end
    wrapper_con_jac = function(Z, rows, cols, vals)
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z)
        end
        nothing
    end
    wrapper_lag_hess = function(Z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z, λ, cost_scaling)
        end
        nothing
    end

    n = trajectory_length * 7
    m = length(callbacks.constraints_lb)
    prob = Ipopt.CreateIpoptProblem(
        n,
        fill(-Inf, n),
        fill(Inf, n),
        m,
        callbacks.constraints_lb,
        callbacks.constraints_ub,
        length(callbacks.full_constraint_jac_triplet.jac_rows),
        length(callbacks.full_lag_hess_triplet.hess_rows),
        wrapper_f,
        wrapper_con,
        wrapper_grad_f,
        wrapper_con_jac,
        wrapper_lag_hess
    )

    controls = repeat([zeros(2),], trajectory_length)
    states = repeat([X¹,], trajectory_length)
    Z_init = compose_trajectory(states, controls)
    prob.x = Z_init

    Ipopt.AddIpoptIntOption(prob, "print_level", 0)
    status = Ipopt.IpoptSolve(prob)
    if status != 0
        @warn "Problem not cleanly solved. IPOPT status is $(status)."
    end
    
    states, controls = decompose_trajectory(prob.x)
    (; states, controls, status)
end
