function callback_generator(trajectory_length, timestep, R)
    # Define symbolic variables for all inputs, as well as trajectory
    # X¹, X², X³, S¹, S², S³, s, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ = let
    #     @variables(X¹[1:4], s[1:4], c[1:4], r[1:4], a[1:8], b[1:4], Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    # end

    # X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z = let
    #     @variables(X¹[1:4], s[1:trajectory_length], t[1:trajectory_length], aₗ[1:2*trajectory_length], bₗ[1:trajectory_length], aᵣ[1:2*trajectory_length], bᵣ[1:trajectory_length], c[1:2*trajectory_length], rᵢ[1:trajectory_length], rₒ[1:trajectory_length], Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    # end

    X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z = let
        @variables(X¹[1:4], s, t, aₗ[1:2], bₗ, aᵣ[1:2], bᵣ, c[1:2], rᵢ, rₒ, Z[1:6*trajectory_length]) .|> Symbolics.scalarize
    end

    states, controls = decompose_trajectory(Z)
    all_states = [[X¹,]; states]

    cost_val = sum(stage_cost(x, u, R) for (x, u) in zip(states, controls))
    cost_grad = Symbolics.gradient(cost_val, Z)

    constraints_val = Symbolics.Num[]
    constraints_lb = Float64[]
    constraints_ub = Float64[]
    for k = 1:trajectory_length
        # if isready(id_channel)
        #     segment_id = take!(id_channel)
        # end
        # segment_index = path_index[segment_id]

        # # evolve state
        append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
        append!(constraints_lb, zeros(4))
        append!(constraints_ub, zeros(4))

        # append!(constraints_val, lane_constraint(states[k], s[k], t[k], aₗ[(k-1)*2+1:2*k], bₗ[k], c[(k-1)*2+1:2*k], rᵢ[k]))
        # append!(constraints_val, lane_constraint(states[k], s[k], t[k], aᵣ[(k-1)*2+1:2*k], bᵣ[k], c[(k-1)*2+1:2*k], rₒ[k]))
        append!(constraints_val, lane_constraint(states[k], s, t, aₗ, bₗ, c, rᵢ))
        append!(constraints_val, lane_constraint(states[k], s, t, aᵣ, bᵣ, c, rₒ))
        append!(constraints_lb, [0.0; -Inf])
        append!(constraints_ub, [Inf; 0.0])

        append!(constraints_val, states[k][3])
        append!(constraints_lb, 0.0)
        append!(constraints_ub, 8.0)

        append!(constraints_val, states[k][4] - X¹[4])
        append!(constraints_lb, -pi/2)
        append!(constraints_ub, pi/2)

        # calculate segment
        # segment_id = find_segment(states[k][1:2], map_segments, keys(path_index))
        # segment_index = path_index[current_segment]

        # if isapprox(map_segments[segment_id].lane_boundaries[1].curvature, 0.0, atol=1e-6)
        #     append!(constraints_val, straight_lane_constraint(states[k], aₗ[(segment_index - 1) * 2 + 1:segment_index * 2], bₗ[segment_index]))
        #     append!(constraints_val, straight_lane_constraint(states[k], aᵣ[(segment_index - 1) * 2 + 1:segment_index * 2], bᵣ[segment_index]))
        #     append!(constraints_lb, [0.0; 0.0])
        #     append!(constraints_ub, [Inf; Inf])
        # else
        #     append!(constraints_val, turn_lane_constraint(states[k], c[(segment_index - 1) * 2 + 1:segment_index * 2], rᵢ[segment_index]))
        #     append!(constraints_val, turn_lane_constraint(states[k], c[(segment_index - 1) * 2 + 1:segment_index * 2], rₒ[segment_index]))
        #     append!(constraints_lb, [0.0; -Inf])
        #     append!(constraints_ub, [Inf; 0.0])
        # end
        
        

        # append!(constraints_val, lane_constraint(states[k], c[1:2], r[1]) + s[2])
        # append!(constraints_val, lane_constraint(states[k], c[1:2], r[2]) - s[2])
        # append!(constraints_lb, [0.0; -Inf])
        # append!(constraints_ub, [Inf; 0.0])
        # append!(constraints_val, straight_lane_constraint(states[k], a[1:2], b[1]) + s[1])
        # append!(constraints_val, straight_lane_constraint(states[k], a[3:4], b[2]) + s[1])
        # append!(constraints_lb, [0.0; 0.0])
        # append!(constraints_ub, [Inf; Inf])
        # append!(constraints_val, collision_constraint(states[k], vehicle_2_prediction[k], r¹ + 1.5, r²)) # Added 1 to radius of car
        # append!(constraints_val, collision_constraint(states[k], vehicle_3_prediction[k], r¹ + 1.5, r³)) # to keep a little more space
        # append!(constraints_lb, zeros(2))
        # append!(constraints_ub, fill(Inf, 2))
        # append!(constraints_val, states[k][4] - X¹[4]) # constrains relative turning angle
        # append!(constraints_lb, -pi/4)
        # append!(constraints_ub, pi/4)
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
        cost_fn = Symbolics.build_function(cost_val, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z]; expression)
        (X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z) -> cost_fn([X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z])
    end

    full_cost_grad_fn = let
        cost_grad_fn! = Symbolics.build_function(cost_grad, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z]; expression)[2]
        (grad, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z) -> cost_grad_fn!(grad, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z])
    end

    full_constraint_fn = let
        constraint_fn! = Symbolics.build_function(constraints_val, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z]; expression)[2]
        (cons, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z) -> constraint_fn!(cons, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z])
    end

    full_constraint_jac_vals_fn = let
        constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z]; expression)[2]
        (vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z) -> constraint_jac_vals_fn!(vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z])
    end
    
    full_hess_vals_fn = let
        hess_vals_fn! = Symbolics.build_function(hess_vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z; λ; cost_scaling]; expression)[2]
        (vals, X¹, s, t, aₗ, bₗ, aᵣ, bᵣ, c, rᵢ, rₒ, Z, λ, cost_scaling) -> hess_vals_fn!(vals, [X¹; s; t; aₗ; bₗ; aᵣ; bᵣ; c; rᵢ; rₒ; Z; λ; cost_scaling])
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
    K = Int(length(Z) / 6)
    controls = [@view(Z[(k - 1) * 2 + 1:k * 2]) for k = 1:K]
    states = [@view(Z[2 * K + (k - 1) * 4 + 1:2 * K + k * 4]) for k = 1:K]
    return states, controls
end

function compose_trajectory(states, controls)
    K = length(states)
    Z = [reduce(vcat, controls); reduce(vcat, states)]
end

function stage_cost(X, U, R)
    cost = -0.1 * X[3] + U' * R * U
end

function evolve_state(X, U, Δ)
    V = X[3] + Δ * U[1]
    θ = X[4] + Δ * U[2]
    X + Δ * [V * cos(θ), V * sin(θ), U[1], U[2]]
end

function inside_segment(X, )

end

function update_controls(X, U, Δ)
    v = X[3] + U[1] * Δ
    ω = X[4] + U[2] * Δ
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



"""
Create functions which accepts X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², as input, and each return
one of the 5 callbacks which constitute an IPOPT problem: 
1. eval_f
2. eval_g
3. eval_grad_f
4. eval_jac_g
5. eval_hess_lag

Xⁱ is the vehicle_state of vehicle i at the start of the trajectory (t=0)
rⁱ is the radius of the i-th vehicle.
(aⁱ, bⁱ) define a halfpsace representing one of the two lane boundaries. 

The purpose of this function is to construct functions which can quickly turn 
updated world information into planning problems that IPOPT can solve.
"""
# function create_callback_generator(trajectory_length=40, timestep=0.2, R = Diagonal([0.1, 0.5]), max_vel=10.0)
#     # Define symbolic variables for all inputs, as well as trajectory
#     X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², Z = let
#         @variables(X¹[1:4], X²[1:4], X³[1:4], r¹, r², r³, a¹[1:2], b¹, a²[1:2], b², Z[1:6*trajectory_length]) .|> Symbolics.scalarize
#     end
   
#     states, controls = decompose_trajectory(Z)
#     all_states = [[X¹,]; states]
#     # vehicle_2_prediction = constant_velocity_prediction(X², trajectory_length, timestep)
#     # vehicle_3_prediction = constant_velocity_prediction(X³, trajectory_length, timestep)
#     cost_val = sum(stage_cost(x, u, R) for (x,u) in zip(states, controls))
#     cost_grad = Symbolics.gradient(cost_val, Z)

#     constraints_val = Symbolics.Num[]
#     constraints_lb = Float64[]
#     constraints_ub = Float64[]
#     for k in 1:trajectory_length
#         append!(constraints_val, all_states[k+1] .- evolve_state(all_states[k], controls[k], timestep))
#         append!(constraints_lb, zeros(4))
#         append!(constraints_ub, zeros(4))
#         append!(constraints_val, lane_constraint(states[k], a¹, b¹, r¹))
#         append!(constraints_val, lane_constraint(states[k], a², b², r¹))
#         append!(constraints_lb, zeros(2))
#         append!(constraints_ub, fill(Inf, 2))
#         append!(constraints_val, collision_constraint(states[k], vehicle_2_prediction[k], r¹, r²))
#         append!(constraints_val, collision_constraint(states[k], vehicle_3_prediction[k], r¹, r³))
#         append!(constraints_lb, zeros(2))
#         append!(constraints_ub, fill(Inf, 2))
#         append!(constraints_val, states[k][3])
#         append!(constraints_lb, 0.0)
#         append!(constraints_ub, max_vel)
#         append!(constraints_val, states[k][4])
#         append!(constraints_lb, -pi/4)
#         append!(constraints_ub, pi/4)
#     end

#     constraints_jac = Symbolics.sparsejacobian(constraints_val, Z)
#     (jac_rows, jac_cols, jac_vals) = findnz(constraints_jac)
#     num_constraints = length(constraints_val)

#     λ, cost_scaling = let
#         @variables(λ[1:num_constraints], cost_scaling) .|> Symbolics.scalarize
#     end

#     lag = (cost_scaling * cost_val + λ' * constraints_val)
#     lag_grad = Symbolics.gradient(lag, Z)
#     lag_hess = Symbolics.sparsejacobian(lag_grad, Z)
#     (hess_rows, hess_cols, hess_vals) = findnz(lag_hess)
    
#     expression = Val{false}

#     full_cost_fn = let
#         cost_fn = Symbolics.build_function(cost_val, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²]; expression)
#         (Z, X¹, X², X³, r¹, r², r³, a¹, b¹, a², b²) -> cost_fn([Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²])
#     end

#     full_cost_grad_fn = let
#         cost_grad_fn! = Symbolics.build_function(cost_grad, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²]; expression)[2]
#         (grad, Z, X¹, X², X³, r¹, r², r³, a¹, b¹, a², b²) -> cost_grad_fn!(grad, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²])
#     end

#     full_constraint_fn = let
#         constraint_fn! = Symbolics.build_function(constraints_val, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²]; expression)[2]
#         (cons, Z, X¹, X², X³, r¹, r², r³, a¹, b¹, a², b²) -> constraint_fn!(cons, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²])
#     end

#     full_constraint_jac_vals_fn = let
#         constraint_jac_vals_fn! = Symbolics.build_function(jac_vals, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²]; expression)[2]
#         (vals, Z, X¹, X², X³, r¹, r², r³, a¹, b¹, a², b²) -> constraint_jac_vals_fn!(vals, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²])
#     end
    
#     full_hess_vals_fn = let
#         hess_vals_fn! = Symbolics.build_function(hess_vals, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²; λ; cost_scaling]; expression)[2]
#         (vals, Z, X¹, X², X³, r¹, r², r³, a¹, b¹, a², b², λ, cost_scaling) -> hess_vals_fn!(vals, [Z; X¹; X²; X³; r¹; r²; r³; a¹; b¹; a²; b²; λ; cost_scaling])
#     end

#     full_constraint_jac_triplet = (; jac_rows, jac_cols, full_constraint_jac_vals_fn)
#     full_lag_hess_triplet = (; hess_rows, hess_cols, full_hess_vals_fn)

#     return (; full_cost_fn, 
#             full_cost_grad_fn, 
#             full_constraint_fn, 
#             full_constraint_jac_triplet, 
#             full_lag_hess_triplet,
#             constraints_lb,
#             constraints_ub)
# end

"""
Predict a dummy trajectory for other vehicles.
"""
# function constant_velocity_prediction(X0, trajectory_length, timestep)
#     X = X0
#     U = zeros(2)
#     states = []
#     for k = 1:trajectory_length
#         X = evolve_state(X, U, timestep)
#         push!(states, X)
#     end
#     states
# end

"""
The physics model used for motion planning purposes.
Returns X[k] when inputs are X[k-1] and U[k]. 
Uses a slightly different vehicle model than presented in class for technical reasons.
"""
# function evolve_state(X, U, Δ)
#     V = X[3] + Δ * U[1] 
#     θ = X[4] + Δ * U[2]
#     X + Δ * [V*cos(θ), V*sin(θ), U[1], U[2]]
# end

# function lane_constraint(X, a, b, r)
#     a'*(X[1:2] - a*r)-b
# end

# function collision_constraint(X1, X2, r1, r2)
#     (X1[1:2]-X2[1:2])'*(X1[1:2]-X2[1:2]) - (r1+r2)^2
# end

"""
Cost at each stage of the plan
"""
# function stage_cost(X, U, R)
#     cost = -0.1 * X[3] + U' * R * U
# end


"""
Assume z = [U[1];...;U[K];X[1];...;X[K]]
Return states = [X[1], X[2],..., X[K]], controls = [U[1],...,U[K]]
where K = trajectory_length
"""
# function decompose_trajectory(z)
#     K = Int(length(z) / 6)
#     controls = [@view(z[(k-1)*2+1:k*2]) for k = 1:K]
#     states = [@view(z[2K+(k-1)*4+1:2K+k*4]) for k = 1:K]
#     return states, controls
# end

# function compose_trajectory(states, controls)
#     K = length(states)
#     z = [reduce(vcat, controls); reduce(vcat, states)]
# end


