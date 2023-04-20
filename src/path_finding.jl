struct HalfSpace
    a::SVector{2,Float64}
    b::Float64
end

function find_segment(position, map_segments, sub_map=keys(map_segments))
    for id in sub_map
        if inside_segment(position, map_segments[id])
            return id
        end
    end
end

function inside_segment(position, segment)
    if segment.lane_boundaries[1].curvature == 0
        return all([h.a' * position > h.b for h in straight_halfspaces(segment)])
    else
        center = center_of_curve(segment)
        inner_radius, outer_radius = radii(segment)
        between_radii = inner_radius < norm(position - center) < outer_radius

        turn_direction = sign(segment.lane_boundaries[1].curvature)
        current_angle = angle_(position, center) * turn_direction
        start_angle = angle_(segment.lane_boundaries[1].pt_a, center) * turn_direction
        end_angle = angle_(segment.lane_boundaries[1].pt_b, center) * turn_direction
        if start_angle > end_angle
            between_angles = start_angle < current_angle < 2 * π || 0 < current_angle < end_angle
        else
            between_angles = start_angle < current_angle < end_angle
        end

        return between_radii && between_angles
    end
end

function straight_halfspaces(segment)
    num_lane_boundaries = length(segment.lane_boundaries)
    hₗ = halfspace(segment.lane_boundaries[1].pt_a, segment.lane_boundaries[1].pt_b)
    hₑ = halfspace(segment.lane_boundaries[1].pt_b, segment.lane_boundaries[num_lane_boundaries].pt_b)
    hᵣ = halfspace(segment.lane_boundaries[num_lane_boundaries].pt_b, segment.lane_boundaries[num_lane_boundaries].pt_a)
    hₛ = halfspace(segment.lane_boundaries[num_lane_boundaries].pt_a, segment.lane_boundaries[1].pt_a)
    [hₗ, hₑ, hᵣ, hₛ]
end

function halfspace(pt_a, pt_b)
    a = [0 1; -1 0] * (pt_b - pt_a)
    a = a / norm(a)
    b = a' * pt_a
    HalfSpace(SVector{2}(a), b)
end

function center_of_curve(segment)
    pt_a = segment.lane_boundaries[1].pt_a
    pt_b = segment.lane_boundaries[1].pt_b
    radius = 1 / segment.lane_boundaries[1].curvature
    mid_chord = midpoint(pt_a, pt_b)
    diag1 = norm(mid_chord - pt_a)
    diag2 = sqrt(radius^2 - diag1^2)
    s = SVector(-1.0, 1.0) * sign(radius) * sign(mid_chord[1] - pt_a[1]) * sign(mid_chord[2] - pt_a[2])
    mid_chord + s .* (diag1 * (mid_chord - pt_a) / diag2)
end

function radii(segment)
    curve = segment.lane_boundaries[1].curvature
    if curve == 0
        return 0.0, 0.0
    end
    num_lane_boundaries = length(segment.lane_boundaries)
    r1 = 1 / abs(curve)
    r2 = 1 / abs(segment.lane_boundaries[num_lane_boundaries].curvature)
    min(r1,r2), max(r1,r2)
end

function angle_(p, c)
    p̂ = (p - c) / norm(p - c)
    θ = atan(p̂[2], p̂[1])
    θ = θ > 0 ? θ : θ + 2 * π
end

function segment_length(segment)
    if segment.lane_boundaries[1].curvature == 0
        return norm(end_midpoint(segment) - midpoint(segment.lane_boundaries[1].pt_a, segment.lane_boundaries[2].pt_a))
    else
        center = center_of_curve(segment)
        inner_radius, outer_radius = radii(segment)

        turn_direction = sign(segment.lane_boundaries[1].curvature)
        start_angle = angle_(segment.lane_boundaries[1].pt_a, center) * turn_direction
        end_angle = angle_(segment.lane_boundaries[1].pt_b, center) * turn_direction
        if end_angle < start_angle
            end_angle += 2 * π
        end

        return (end_angle - start_angle) * (outer_radius + inner_radius) / 2
    end
end

function midpoint(a::SVector{2,Float64}, b::SVector{2,Float64})
    a + (b - a) / 2
end

function end_midpoint(segment)
    midpoint(segment.lane_boundaries[1].pt_b, segment.lane_boundaries[2].pt_b)
end

function target_point(segment)
    num_lane_bounds = length(segment.lane_boundaries)
    if segment.lane_boundaries[1].curvature == 0
        mid_start = midpoint(segment.lane_boundaries[num_lane_bounds - 1].pt_a, segment.lane_boundaries[num_lane_bounds].pt_a)
        mid_end = midpoint(segment.lane_boundaries[num_lane_bounds - 1].pt_b, segment.lane_boundaries[num_lane_bounds].pt_b)
        return midpoint(mid_start, mid_end)
    else
        center = center_of_curve(segment)
        inner_radius, outer_radius = radii(segment)
        mid = midpoint(segment.lane_boundaries[1].pt_a, segment.lane_boundaries[1].pt_b)
        return center + (mid - center) / norm(mid - center) * (outer_radius + inner_radius) / 2
    end
end

function heuristic(map_segments, segment_id, target_id)
    norm(target_point(map_segments[target_id]) - end_midpoint(map_segments[segment_id]))
end

function cost_function(map_segments, segment_id)
    segment_length(map_segments[segment_id])
end

function a_star(map_segments, start_id, target_id)
    open_set = PriorityQueue{Int, Float64}()
    enqueue!(open_set, start_id, 0.0)

    g_scores = Dict{Int, Float64}()
    f_scores = Dict{Int, Float64}()
    h_scores = Dict{Int, Float64}()
    g_scores[start_id] = 0.0
    h_scores[start_id] = heuristic(map_segments, start_id, target_id)
    f_scores[start_id] = h_scores[start_id]
    ancestors = Dict{Int, Int}()

    while !isempty(open_set)
        current_id = dequeue!(open_set)

        if current_id == target_id
            return construct_path(ancestors, start_id, target_id)
        end

        for child_id in map_segments[current_id].children
            tentative_g_score = g_scores[current_id] + cost_function(map_segments, child_id)
            
            if !(child_id in keys(g_scores)) || tentative_g_score < g_scores[child_id]
                g_scores[child_id] = tentative_g_score
                h_scores[child_id] = heuristic(map_segments, child_id, target_id)
                f_scores[child_id] = g_scores[child_id] + h_scores[child_id]
                ancestors[child_id] = current_id

                if !haskey(open_set, child_id)
                    enqueue!(open_set, child_id, f_scores[child_id])
                end
            end
        end
    end

    return [] # Return an empty array if no path is found
end

function construct_path(ancestors, start_id, target_id)
    path = Dict{Int, Int}()
    path_index = OrderedDict{Int, Int}()
    index = 1

    path[target_id] = 0 # no valid child on path
    path_index[target_id] = index

    current_id = target_id
    while current_id != start_id
        child_id = current_id
        current_id = ancestors[child_id]
        index += 1
        path[current_id] = child_id
        path_index[current_id] = index
    end

    path, path_index
end
