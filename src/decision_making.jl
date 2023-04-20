function sim(socket, gt_channel, map_segments, path;
        rng = MersenneTwister(420),
        sim_steps = 100,
        timestep = 0.2,
        trajectory_length = 10,
        R = Diagonal([0.1, 0.5]),
        track_center = [-76.67, -75.0],
        track_radius = 15,
        lane_width = 10, 
        r = 2.0,
        max_vel = 10.0)

    # p1 = -1.0 * track_radius
    # p2 = 0.0 * track_radius
    # v = max_vel
    # θ = π / 2
    # S = r
    # ego = (; state=[p1, p2, v, θ], S)

    # TODO Setup callbacks appropriately
    callbacks = callback_generator(trajectory_length, timestep, R)

    if isready(gt_channel)
        gt_meas = fetch(gt_channel)
    end

    # cmd = VehicleCommand(0.0, 10.0, true)
    # serialize(socket, cmd)

    @showprogress for t = 1:sim_steps
        while isready(gt_channel)
            meas = take!(gt_channel)
            if meas.time > gt_meas.time
                gt_meas = meas
            end
        end

        ego = (; state=[gt_meas.position[1:2]; gt_meas.velocity[1]; gt_meas.angular_velocity[1]], S=r)

        s0 = find_segment(ego.state[1:2], map_segments, keys(path))
        s1 = path[s0]

        s = zeros(4)
        s0_curve = map_segments[s0].lane_boundaries[1].curvature
        s1_curve = map_segments[s1].lane_boundaries[1].curvature
        if (s0_curve == 0)
            s[2] = 100000
        else
            s[1] = 100000
        end
        if (s1_curve == 0)
            s[4] = 100000
        else
            s[3] = 100000
        end

        h1 = halfspace(map_segments[s0].lane_boundaries[1].pt_a, map_segments[s0].lane_boundaries[1].pt_b)
        h2 = halfspace(map_segments[s0].lane_boundaries[2].pt_b, map_segments[s0].lane_boundaries[2].pt_a)
        h3 = halfspace(map_segments[s1].lane_boundaries[1].pt_a, map_segments[s1].lane_boundaries[1].pt_b)
        h4 = halfspace(map_segments[s1].lane_boundaries[2].pt_b, map_segments[s1].lane_boundaries[2].pt_a)
        a = [h1.a; h2.a; h3.a; h4.a]
        b = [h1.b; h2.b; h3.b; h4.b]

        c1 = center_of_curve(map_segments[s0])
        c2 = center_of_curve(map_segments[s1])
        ri1, ro1 = radii(map_segments[s0])
        ri2, ro2 = radii(map_segments[s1])
        c = [c1; c2]
        r = [ri1; ro1; ri2; ro2]

        @info(s0, " ", s1)
        @info("X = ", ego.state[1:2])
        @info("s = ", s)
        @info("a = ", a)
        @info("b = ", b)
        @info("c = ", c)
        @info("r = ", r)

        trajectory = generate_trajectory(ego, s, c, r, a, b, callbacks, trajectory_length, timestep)
        # @info(trajectory.controls[1])
        velocity, steering_angle = update_controls(ego.state, trajectory.controls[1], timestep)

        cmd = VehicleCommand(steering_angle, velocity, true)
        serialize(socket, cmd)
    end

    #cmd = VehicleCommand(0.0, 0.0, false)
    #serialize(socket, cmd)
end

function generate_trajectory(ego, s, c, r, a, b, callbacks, trajectory_length, timestep)
    X¹ = ego.state
    S¹ = ego.S

    wrapper_f = function(Z)
        callbacks.full_cost_fn(Z, X¹, S¹, s, c, r, a, b)
    end
    wrapper_grad_f = function(Z, grad)
        callbacks.full_cost_grad_fn(grad, Z, X¹, S¹, s, c, r, a, b)
    end
    wrapper_con = function(Z, con)
        callbacks.full_constraint_fn(con, Z, X¹, S¹, s, c, r, a, b)
    end
    wrapper_con_jac = function(Z, rows, cols, vals)
        if isnothing(vals)
            rows .= callbacks.full_constraint_jac_triplet.jac_rows
            cols .= callbacks.full_constraint_jac_triplet.jac_cols
        else
            callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, Z, X¹, S¹, s, c, r, a, b)
        end
        nothing
    end
    wrapper_lag_hess = function(Z, rows, cols, cost_scaling, λ, vals)
        if isnothing(vals)
            rows .= callbacks.full_lag_hess_triplet.hess_rows
            cols .= callbacks.full_lag_hess_triplet.hess_cols
        else
            callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, Z, X¹, S¹, s, c, r, a, b, λ, cost_scaling)
        end
        nothing
    end

    n = trajectory_length * 6
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

function simulate(;
        rng = MersenneTwister(420),
        sim_steps = 100,
        timestep = 0.2,
        trajectory_length = 10,
        R = Diagonal([0.1, 0.5]),
        track_center = [0.0, 0.0],
        track_radius = 15,
        lane_width = 10, 
        r = 2.0,
        max_vel = 10.0)

    sim_records = []
    # track_center = [-76.67, -75.0]
    
    # vehicle 1 is EGO
    # vehicles = [generate_random_vehicle(rng, lane_width, track_radius, track_center, min_r, max_r, max_vel),]
    # while length(vehicles) < num_vehicles
    #     v = generate_random_vehicle(rng, lane_width, track_radius, track_center, min_r, max_r, max_vel/2)
    #     if any(collision_constraint(v.state, v2.state, v.r, v2.r) < 1.0 for v2 in vehicles)
    #         continue
    #     else
    #         push!(vehicles, v)
    #     end
    # end
    # target_radii = [norm(v.state[1:2]-track_center) for v in vehicles]

    p1 = -1.0 * track_radius
    p2 = 0.0 * track_radius
    v = max_vel
    θ = π / 2
    S = r
    ego = (; state=[p1, p2, v, θ], S)

    # TODO Setup callbacks appropriately
    callbacks = callback_generator(trajectory_length, timestep, R)

    @showprogress for t = 1:sim_steps
        # ego = vehicles[1]
        # dists = [Inf; [norm(v.state[1:2]-ego.state[1:2])-v.r-ego.r for v in vehicles[2:end]]]
        # closest = partialsortperm(dists, 1:2)
        # V2 = vehicles[closest[1]]
        # V3 = vehicles[closest[2]]
        
        trajectory = generate_trajectory(ego, track_center, track_radius, lane_width, callbacks, trajectory_length, timestep)
        
        push!(sim_records, (; vehicles=[ego], trajectory))
        ego = (; state=trajectory.states[1], S=r)

        # for i in 2:num_vehicles
        #     old_state = vehicles[i].state
        #     control = generate_tracking_control(target_radii[i], track_center, old_state)
        #     new_state = evolve_state(old_state, control, timestep)
        #     vehicles[i] = (; state=new_state, vehicles[i].r)
        # end
        
        #foreach(i->vehicles[i] = (; state = evolve_state(vehicles[i].state, zeros(2), timestep), vehicles[i].r), 2:num_vehicles)
    end
    #visualize_simulation(sim_records, track_radius, track_center, lane_width)
end

# function wrap(X, lane_length)
#     X_wrapped = copy(X)
#     if X_wrapped[1] > lane_length
#         X_wrapped[1] -= lane_length
#     end
#     X_wrapped
# end

# function generate_random_vehicle(lane_width, lane_length, min_r, max_r, max_vel)
#     r = (max_r-min_r) + min_r
#     p1 = lane_length
#     p2 = (lane_width-2r)+(r-lane_width/2)
#     v = max_vel/2 + max_vel
#     θ = 0.0
#     (; state=[p1, p2, v, θ], r)
# end

# function generate_trajectory(ego, V2, V3, a¹, b¹, a², b², callbacks, trajectory_length, timestep)
#     X1 = ego.state
#     X2 = V2.state
#     X3 = V3.state
#     r1 = ego.r
#     r2 = V2.r
#     r3 = V3.r

#     # refine callbacks with current values of parameters / problem inputs
#     wrapper_f = function(z)
#         callbacks.full_cost_fn(z, X1, X2, X3, r1, r2, r3, a¹, b¹, a², b²)
#     end
#     wrapper_grad_f = function(z, grad)
#         callbacks.full_cost_grad_fn(grad, z, X1, X2, X3, r1, r2, r3, a¹, b¹, a², b²)
#     end
#     wrapper_con = function(z, con)
#         callbacks.full_constraint_fn(con, z, X1, X2, X3, r1, r2, r3, a¹, b¹, a², b²)
#     end
#     wrapper_con_jac = function(z, rows, cols, vals)
#         if isnothing(vals)
#             rows .= callbacks.full_constraint_jac_triplet.jac_rows
#             cols .= callbacks.full_constraint_jac_triplet.jac_cols
#         else
#             callbacks.full_constraint_jac_triplet.full_constraint_jac_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, a¹, b¹, a², b²)
#         end
#         nothing
#     end
#     wrapper_lag_hess = function(z, rows, cols, cost_scaling, λ, vals)
#         if isnothing(vals)
#             rows .= callbacks.full_lag_hess_triplet.hess_rows
#             cols .= callbacks.full_lag_hess_triplet.hess_cols
#         els
#             callbacks.full_lag_hess_triplet.full_hess_vals_fn(vals, z, X1, X2, X3, r1, r2, r3, a¹, b¹, a², b², λ, cost_scaling)
#         end
#         nothing
#     end

#     n = trajectory_length*6
#     m = length(callbacks.constraints_lb)
#     prob = Ipopt.CreateIpoptProblem(
#         n,
#         fill(-Inf, n),
#         fill(Inf, n),
#         length(callbacks.constraints_lb),
#         callbacks.constraints_lb,
#         callbacks.constraints_ub,
#         length(callbacks.full_constraint_jac_triplet.jac_rows),
#         length(callbacks.full_lag_hess_triplet.hess_rows),
#         wrapper_f,
#         wrapper_con,
#         wrapper_grad_f,
#         wrapper_con_jac,
#         wrapper_lag_hess
#     )

#     controls = repeat([zeros(2),], trajectory_length)
#     #states = constant_velocity_prediction(X1, trajectory_length, timestep)
#     states = repeat([X1,], trajectory_length)
#     zinit = compose_trajectory(states, controls)
#     prob.x = zinit

#     Ipopt.AddIpoptIntOption(prob, "print_level", 0)
#     status = Ipopt.IpoptSolve(prob)

#     if status != 0
#         @warn "Problem not cleanly solved. IPOPT status is $(status)."
#     end
#     states, controls = decompose_trajectory(prob.x)
#     (; states, controls, status)
# end

function visualize_simulation(sim_results, track_radius, track_center, lane_width)
    f = Figure()
    ax = f[1,1] = Axis(f, aspect = DataAspect())
    r_inside = track_radius - lane_width/2
    r_outside = track_radius + lane_width/2
    xlims!(ax, track_center[1]-r_outside, track_center[1]+r_outside)
    ylims!(ax, track_center[2]-r_outside, track_center[2]+r_outside)

    θ = 0:0.1:2pi+0.1
    lines!(r_inside*cos.(θ), r_inside*sin.(θ), color=:black, linewidth=3)
    lines!(r_outside*cos.(θ), r_outside*sin.(θ), color=:black, linewidth=3)

    ps = [Observable(Point2f(v.state[1], v.state[2])) for v in sim_results[1].vehicles]
    traj = [Observable(Point2f(state[1], state[2])) for state in sim_results[1].trajectory.states]
    for t in traj
        plot!(ax, t, color=:green)
    end

    circles = [@lift(Circle($p, v.S)) for (p,v) in zip(ps, sim_results[1].vehicles)]
    for (e, circle) in enumerate(circles)
        if e == 1
            poly!(ax, circle, color = :blue)
        else
            poly!(ax, circle, color = :red)
        end
    end
    # record(f, "mpc_circle_animation.mp4", sim_results; framerate = 10) do sim_step
    #(uncomment above line and comment below line to create visualization)
    for sim_step in sim_results
        for (t,state) in zip(traj, sim_step.trajectory.states)
            t[] = Point2f(state[1], state[2])
        end
        for (p,v) in zip(ps, sim_step.vehicles)
            p[] = Point2f(v.state[1], v.state[2])
        end
        display(f)
        sleep(0.25)
    end
end
