function callback_generator(trajectory_length, timestep, R)
    # Define symbolic variables for all inputs, as well as trajectory
    X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z = let
        @variables(X¹[1:5], s[1:trajectory_length], t[1:trajectory_length], aₗ[1:2*trajectory_length], bₗ[1:trajectory_length], aᵣ[1:2*trajectory_length], bᵣ[1:trajectory_length], c[1:2*trajectory_length], rᵢ[1:trajectory_length], rₒ[1:trajectory_length], aₜ[1:2], bₜ, Z[1:7*trajectory_length]) .|> Symbolics.scalarize
    end

    states, controls = decompose_trajectory(Z)
    all_states = [[X¹,]; states]

    cost_val = sum(stage_cost(x, u, R, aₜ, bₜ) for (x, u) in zip(states, controls))
    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]
    for k = 1:trajectory_length
        # Evolve state constraint
        append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(5))
        append!(constraints_ub, zeros(5))

        # Lane constraints using min/max corners
        # corner11 = lane_constraint(states[k][1:2] + 6.6 * [cos(states[k][3]), sin(states[k][3])] + 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aₗ[(k-1)*2+1:2*k], bₗ[k], c[(k-1)*2+1:2*k], rᵢ[k])
        # corner12 = lane_constraint(states[k][1:2] + 6.6 * [cos(states[k][3]), sin(states[k][3])] + 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k])
        # corner21 = lane_constraint(states[k][1:2] + 6.6 * [cos(states[k][3]), sin(states[k][3])] - 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aₗ[(k-1)*2+1:2*k], bₗ[k], c[(k-1)*2+1:2*k], rᵢ[k])
        # corner22 = lane_constraint(states[k][1:2] + 6.6 * [cos(states[k][3]), sin(states[k][3])] - 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k])
        # corner31 = lane_constraint(states[k][1:2] - 6.6 * [cos(states[k][3]), sin(states[k][3])] + 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aₗ[(k-1)*2+1:2*k], bₗ[k], c[(k-1)*2+1:2*k], rᵢ[k])
        # corner32 = lane_constraint(states[k][1:2] - 6.6 * [cos(states[k][3]), sin(states[k][3])] + 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k])
        # corner41 = lane_constraint(states[k][1:2] - 6.6 * [cos(states[k][3]), sin(states[k][3])] - 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aₗ[(k-1)*2+1:2*k], bₗ[k], c[(k-1)*2+1:2*k], rᵢ[k])
        # corner42 = lane_constraint(states[k][1:2] - 6.6 * [cos(states[k][3]), sin(states[k][3])] - 2.85 * [sin(states[k][3]), cos(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k])

        # append!(constraints_val, min(corner11, corner21, corner31, corner41))
        # append!(constraints_val, max(corner12, corner22, corner32, corner42))
        # append!(constraints_lb, [0.0; -Inf])
        # append!(constraints_ub, [Inf; 0.0])

        # Lane constraints on center of vehicle
        append!(constraints_val, lane_constraint(states[k][1:2] + 4 * [cos(states[k][3]), sin(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k]))
        append!(constraints_val, lane_constraint(states[k][1:2] + 4 * [cos(states[k][3]), sin(states[k][3])], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k]))
        append!(constraints_lb, [0.0; -Inf])
        append!(constraints_ub, [Inf; 0.0])

        # Velocity constraint
        append!(constraints_val, states[k][4])
        append!(constraints_lb, 0.0)
        append!(constraints_ub, 10.0)

        # Slow down for turn constraint
        append!(constraints_val, states[k][4] - distance_to_turn(states[k], aₜ, bₜ))
        append!(constraints_lb, -Inf)
        append!(constraints_ub, 2.0)

        # Angular velocity constraint
        append!(constraints_val, states[k][5])
        append!(constraints_lb, -0.5)
        append!(constraints_ub, 0.5)

        # append!(constraints_val, controls[k][1])
        # append!(constraints_lb, -2.0)
        # append!(constraints_ub, 2.0)

        # append!(constraints_val, controls[k][2])
        # append!(constraints_lb, -5.0)
        # append!(constraints_ub, 5.0)

    end

    constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
    (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
    num_constraints = length(constraints_val)

    λ, cost_scaling = let
        @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
    end

    lag = (cost_scaling * cost_val + λ' * constraints_val)
    lag_grad = Symbolics.gradient(lag, Z)
    lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
    (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
    expression = Val{false}

    full_cost_fn = let
        cost_fn = Symbolics.build_function(cost_val, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z]; expression)
        (X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z) -> cost_fn([X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z]; expression)[2]
        (grad, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z) -> cost_grad_fn!(grad, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z])
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z]; expression)[2]
        (cons, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z) -> constraint_fn!(cons, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z]; expression)[2]
        (vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z) -> constraint_jac_vals_fn!(vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z; λ; cost_scaling]; expression)[2]
        (vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, aₜ, bₜ, Z, λ, cost_scaling) -> hess_vals_fn!(vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; aₜ; bₜ; Z; λ; cost_scaling])
    end

    full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
    full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)

    return (; full_cost_fn, 
            full_cost_grad_fn, 
            full_constraint_fn, 
            full_constraint_jac_triplet, 
            full_lag_hess_triplet,
            constraints_lb,
            constraints_ub)
end

function decompose_trajectory(Z)
    K = Int(length(Z) / 7)
    controls = [@view(Z[(k - 1) * 2 + 1:k * 2]) for k = 1:K]
    states = [@view(Z[2 * K + (k - 1) * 5 + 1:2 * K + k * 5]) for k = 1:K]
    return states, controls
end

function compose_trajectory(states, controls)
    K = length(states)
    Z = [reduce(vcat, controls); reduce(vcat, states)]
end

function stage_cost(X, U, R, a, b)
    cost = -0.1 * X[4] + U' * R * U #+ 0.1 * distance_to_turn(X, a, b)
end

function evolve_state(X, U, Δ)
    v = X[4] + Δ * U[1]
    ω = X[5] + Δ * U[2]
    θ = X[3] + Δ * ω
    X + Δ * [v * cos(θ), v * sin(θ), ω, U[1], U[2]]
end

function update_controls(X, U, Δ)
    v = X[4] + U[1] * Δ
    ω = X[5] + U[2] * Δ
    v, ω
end

function lane_constraint(X, s, t, a, b, c, r)
    s * straight_lane_constraint(X, a, b) + t * turn_lane_constraint(X, c, r)
end

function straight_lane_constraint(X, a, b)
    a' * X[1:2] - b
end

function turn_lane_constraint(X, c, r)
    norm(X[1:2] - c) - r
end

function distance_to_turn(X, a, b)
    max(straight_lane_constraint(X, a, b), 0)
end
