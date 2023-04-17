struct HalfSpace
    a::SVector{2,Float64}
    b::Float64
end

function find_segment(position, map_segments)
    for (id, segment) in map_segments
        if inside_segment(position, segment)
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
        start_angle = angle_(segment.lane_boundaries[1].pt_a, center)
        end_angle = angle_(segment.lane_boundaries[1].pt_b, center)
        curve = sign(segment.lane_boundaries[1].curvature)
        return inner_radius < norm(position - center) < outer_radius && curve * start_angle < curve * angle_(position, center) < curve * end_angle
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
    r = 1 / segment.lane_boundaries[1].curvature
    midpoint = pt_a + (pt_b - pt_a) / 2
    diag1 = norm(midpoint - pt_a)
    diag2 = sqrt(r^2 - diag1^2)
    s = SVector(1.0, -1.0) * sign(r)
    midpoint + s .* (diag1 * (midpoint - pt_a) / diag2)
end

function radii(segment)
    num_lane_boundaries = length(segment.lane_boundaries)
    r1 = 1 / abs(segment.lane_boundaries[1].curvature)
    r2 = 1 / abs(segment.lane_boundaries[num_lane_boundaries].curvature)
    min(r1,r2), max(r1,r2)
end

function angle_(p, c)
    p̂ = (p - c) / norm(p - c)
    θ = atan(p̂[2], p̂[1])
    θ = θ > 0 ? θ : θ + 2 * π
end

function plot_map(map_segments)
    x = []
    y = []
    for (id, segment) in map_segments
        push!(x, segment.lane_boundaries[1].pt_a[1])
        push!(y, segment.lane_boundaries[1].pt_a[2])
        push!(x, segment.lane_boundaries[1].pt_b[1])
        push!(y, segment.lane_boundaries[1].pt_b[2])
    end

    scatter(x, y, markercolor=:yellow)
end
