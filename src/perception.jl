# The entire perception.jl file is working on ONE bounding box (i.e. only four corners of z is given).
# So, if there are two bounding boxes coming in per vehicle, the ekf function must be called twcie in
# example_project.jl's perception() function.

function perception_f(x, delta_t)
    # update xk = [p1 p2 theta vel l w h]
    # - updated-p1 = p1 + delta_time *cos(theta)*v
    # - updated-p2 = p2 + delta_time*sin(theta)*v
    theta_k = x[3]
    vel_k = x[4]
    result = x + delta_t * [(vel_k * cos(theta_k)), (vel_k * sin(theta_k)), 0, 0, 0, 0, 0]
    return result
end

function perception_jac_fx(x, delta_t)
    theta = x[3]
    vel = x[4]

    [1 0 (-sin(theta)*vel*delta_t) (delta_t*cos(theta)) 0 0 0
        0 1 (cos(theta)*vel*delta_t) (delta_t*sin(theta)) 0 0 0
        0 0 1 0 0 0 0
        0 0 0 1 0 0 0
        0 0 0 0 1 0 0
        0 0 0 0 0 1 0
        0 0 0 0 0 0 1
    ]
end

function perception_get_3d_bbox_corners(x, box_size)
    theta = x[3]
    # referenced L20 pg.4 as well
    R = 0.5 * [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]
    xyz = [x[1], x[2], box_size[3] / 2]
    T = [R xyz]
    corners = []

    for dx in [-box_size[1] / 2, box_size[1] / 2]
        for dy in [-box_size[2] / 2, box_size[2] / 2]
            for dz in [-box_size[3] / 2, box_size[3] / 2]
                push!(corners, T * [dx, dy, dz, 1])
            end
        end
    end
    corners
end

"""
    Usage:
        - should be called per camera

    Parameters:
        - x_other: state of a recognized car (in [p1 p2 theta vel l w h])
        - x_ego: state of ego car (in a format that localization gives)
"""
function perception_h(x_other, x_ego, cam_id)
    # constant variables
    # vehicle_size = SVector(13.2, 5.7, 5.3)
    vehicle_size = [13.2 5.7 5.3]
    focal_len = 0.01
    pixel_len = 0.001
    image_width = 640
    image_height = 480
    # num_vehicles = length(x_other)

    corners_body = [perception_get_3d_bbox_corners(x_other, vehicle_size)]
    num_corners = length(corners_body)

    # Section 1: Get transformation matrices
    T_body_cam1 = get_cam_transform(1)
    T_body_cam2 = get_cam_transform(2)
    T_cam_camrot = get_rotated_camera_transform()

    T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
    T_body_camrot2 = multiply_transforms(T_body_cam2, T_cam_camrot)

    # make sure the ego state you get from localization team follows the same format
    # below is the real code:
    # T_world_body = get_body_transform(x_ego.q[1:4], x_ego.q[5:7]) # get_body_transform(quat, loc)
    # below is for testing/debugging:
    T_world_body = get_body_transform(x_ego[1:4], x_ego[5:7])
    T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
    T_world_camrot2 = multiply_transforms(T_world_body, T_body_camrot2)

    transform = invert_transform(T_world_camrot1) # defulat to camera 1
    if cam_id == 2
        transform = invert_transform(T_world_camrot2)
    end

    # Section 2: Calculate the bounding boxes
    bbox = []

    # NOTE: deal with having only 1 or 2 boxes here
    # similar to x_carrot = R * [q1 q2 q3] + t, turn points rotated cam frame
    corners_of_other_vehicle = zeros(Float64, 8, 3)
    for j = 1:num_corners
        corners_of_other_vehicle = [transform * [pt; 1] for pt in corners_body[j]]
    end
    # display("corners_of_other_vehicle")
    # display(corners_of_other_vehicle)
    # println()

    left = image_width / 2
    right = -image_width / 2
    top = image_height / 2
    bot = -image_height / 2

    # keep track of the 3D points of corner values to use in jacoabian
    top_cnr = [0 0 0]
    left_cnr = [0 0 0]
    bot_cnr = [0 0 0]
    right_cnr = [0 0 0]
    corner_ids = [0 0 0 0] # top, left, bot, right
    iter = 1

    # px_py = []
    # we are basically getting through each corner values in camera frame and 
    # keep updating the left, top, bottom, right values!
    for corner in corners_of_other_vehicle
        # every point of corner in camera frame now
        if corner[3] < focal_len
            break
        end
        # display("getting px and py")
        px = focal_len * corner[1] / corner[3]
        py = focal_len * corner[2] / corner[3]
        # push!(px_py, [px py])

        # update the corners
        if py < top
            top_cnr = corner
            corner_ids[1] = iter
        end
        if px < left
            left_cnr = corner
            corner_ids[2] = iter
        end
        if py > bot
            bot_cnr = corner
            corner_ids[3] = iter
        end
        if px > right
            right_cnr = corner
            corner_ids[4] = iter
        end

        top = min(top, py)
        left = min(left, px)
        bot = max(bot, py)
        right = max(right, px)

        iter = iter + 1
        # display("moving on to next iter")
    end
    corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
    # print(px_py)
    # display("corners done")
    # display(corners)
    # println()

    if top ≈ bot || left ≈ right || top > bot || left > right
        println("returning empty bbox")
        # out of frame - return empty bbox
        return bbox, corner_ids, corners
    else
        # update bbox
        top_u = unrounded_convert_to_pixel(image_height, pixel_len, top)
        bot_u = unrounded_convert_to_pixel(image_height, pixel_len, bot)
        left_u = unrounded_convert_to_pixel(image_width, pixel_len, left)
        right_u = unrounded_convert_to_pixel(image_width, pixel_len, right)
        push!(bbox, SVector(top_u, left_u, bot_u, right_u))
    end
    # println()
    # display("now returning perception_h")
    # println()

    return bbox, corner_ids, corners
end


function calculate_J1_for_jac_hx(corner_id, x_other)
    # In perception_h, corners are set in the following format: 
    # corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
    l_mult = 0
    w_mult = 0
    h_mult = 0
    if corner_id == 1
        l_mult = -1
        w_mult = -1
        h_mult = -1
    elseif corner_id == 2
        l_mult = -1
        w_mult = -1
        h_mult = 1
    elseif corner_id == 3
        l_mult = -1
        w_mult = 1
        h_mult = -1
    elseif corner_id == 4
        l_mult = -1
        w_mult = 1
        h_mult = 1
    elseif corner_id == 5
        l_mult = 1
        w_mult = -1
        h_mult = -1
    elseif corner_id == 6
        l_mult = 1
        w_mult = -1
        h_mult = 1
    elseif corner_id == 7
        l_mult = 1
        w_mult = 1
        h_mult = -1
    else
        l_mult = 1
        w_mult = 1
        h_mult = 1
    end

    θ = x_other[3]
    theta = x_other[3]
    l = 13.2
    w = 5.7
    h = 5.3

    # original
    # J1 = [1 0 (0.5*(-sin(theta)*l_mult*l-cos(theta)*w_mult*w)) 0 (0.5*(cos(theta)*l_mult)) (0.5*(-sin(theta)*w_mult)) 0
    #     0 1 (0.5*(cos(theta)*l_mult*l-sin(theta)*w_mult*w)) 0 (0.5*sin(theta)*l_mult) (0.5*(cos(theta)*w_mult)) 0
    #     0 0 0 0 0 0 (0.5+h_mult*0.5)]

    # with 1/4
    # J1 = [1 0 (1/4*(-sin(theta)*l_mult*l-cos(theta)*w_mult*w)) 0 (1/4*(cos(theta)*l_mult)) (1/4*(-sin(theta)*w_mult)) 0
    #     0 1 (1/4*(cos(theta)*l_mult*l-sin(theta)*w_mult*w)) 0 (1/4*sin(theta)*l_mult) (1/4*(cos(theta)*w_mult)) 0
    #     0 0 0 0 0 0 (h_mult*3/4)]

    # with l, w, h set to 0
    # J1 = [1 0 (1/4*(-sin(theta)*l_mult*l-cos(theta)*w_mult*w)) 0 0 0 0
    #     0 1 (1/4*(cos(theta)*l_mult*l-sin(theta)*w_mult*w)) 0 0 0 0
    #     0 0 0 0 0 0 0]

    # from prof
    J1 = [1 0 l_mult*l/2*(-sin(θ))+l_mult*w/2*(-cos(θ)) 0 cos(θ)/2*l_mult -sin(θ)/2*l_mult 0
        0 1 w_mult*l/2*(cos(θ))+w_mult*w/2*(-sin(θ)) 0 sin(θ)/2*w_mult cos(θ)/2*w_mult 0
        0 0 0 0 0 0 0.5+h_mult/2]
    # println("J1 from mine")
    # println(J1)
    return J1
end

function perception_jac_hx(corner, corner_id, x_other, x_ego, cam_id)
    # Calculate J1
    J1 = calculate_J1_for_jac_hx(corner_id, x_other) # do i even need the corner values..? i can just have cornder_id to figure out l_mult and w_mult and h_mult
    # display("done with J1")

    # Calculate J2 -- confirmed it's correct
    T_body_cam1 = get_cam_transform(1)
    T_body_cam2 = get_cam_transform(2)
    T_cam_camrot = get_rotated_camera_transform()

    T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
    T_body_camrot2 = multiply_transforms(T_body_cam2, T_cam_camrot)

    # make sure the ego state you get from localization team follows the same format
    # below is the real code:
    # T_world_body = get_body_transform(x_ego.q[1:4], x_ego.q[5:7]) # get_body_transform(quat, loc)
    # below is for testing/debugging:
    T_world_body = get_body_transform(x_ego[1:4], x_ego[5:7])
    T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
    T_world_camrot2 = multiply_transforms(T_world_body, T_body_camrot2)

    T_Rt = invert_transform(T_world_camrot1) # defulat to camera 1 - default is 3x4 --> take only the first 3 columns
    if cam_id == 2
        T_Rt = invert_transform(T_world_camrot2)
    end
    J2 = T_Rt[:, 1:3]
    # display("done with J2")

    # Calculate J3
    focal_len = 0.01
    J3 = [focal_len/corner[3] 0 -focal_len*corner[1]/((corner[3])^2)
        0 focal_len/corner[3] -focal_len*corner[2]/((corner[3])^2)]
    # display("done with J3")

    # Calculate J4 -- confirmed it's correct
    pixel_len = 0.001
    s = 1 / pixel_len
    J4 = [s 0; 0 s]
    # display("done with J4")

    return J4 * J3 * J2 * J1
end


"""
    Usage:
        - should be called per camera -- per vehicle is addressed in example_project.jl file
    Variables:
        - x = state of the other car = [p1 p2 theta vel l w h] (7 x 1)
        - P = covariance of the state
        - z = measurement = [y1a y2a y1b y2b] (4 x 1)
        - Q = process noise
        - R = measurement noise
"""
function perception_ekf(x0, bbox, xego, delta_t, cam_id)
    println("starting perception ekf now")
    # constant noise variables
    covariance_p = Diagonal([1^2, 1^2, 0.2^2, 0.4^2, 0.0003^2, 0.0002^2, 0.0001^2])  # covariance for process model
    # covariance_z = Diagonal([0.7^2, 0.7^2, 0.7^2, 0.7^2]) # covariance for measurement model
    println(bbox)

    covariance_z = Diagonal([bbox[1], bbox[2], bbox[3], bbox[4]]) + Diagonal([0.7^2, 0.7^2, 0.7^2, 0.7^2]) # covariance for measurement model

    # mean value of xk state of the OTHER car
    mu = x0
    sigma = Diagonal([1^2, 1^2, 0.2^2, 0.4^2, 0.005^2, 0.003^2, 0.001^2])

    xk = perception_f(mu, delta_t)
    zk, corner_ids, corners = perception_h(xk, xego, cam_id)
    zk = zk[1] # just re-formatting in order to have the right data structure

    println("debug1")
    # *All of the equations below are referenced from L17 pg.3 and HW4
    # Process model: P(xk | xk-1, bbxk) = N(A*x-1, sig_carrot))
    # A = perception_jac_fx(mus[k], delta_t)
    # sigma_carrot = covariance_p + A * sigmas[k] * A'
    # mu_carrot = perception_f(mus[k], delta_t)

    A = perception_jac_fx(mu, delta_t)
    sigma_carrot = covariance_p + A * sigma * A'
    mu_carrot = perception_f(mu, delta_t)

    # println("debug2")

    # Measurement model
    # In perception_h, corners are set in the following format: 
    # corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
    C_top = perception_jac_hx(corners[1], corner_ids[1], mu_carrot, xego, cam_id)
    C_left = perception_jac_hx(corners[2], corner_ids[2], mu_carrot, xego, cam_id)
    C_bot = perception_jac_hx(corners[3], corner_ids[3], mu_carrot, xego, cam_id)
    C_right = perception_jac_hx(corners[4], corner_ids[4], mu_carrot, xego, cam_id)

    # now stack them up to have a 4 x 7 matrix
    # We only care about the top row for left and right
    # We only care about the bottom row for top and bottom
    C = [transpose(C_top[2, :]); transpose(C_left[1, :]); transpose(C_bot[2, :]); transpose(C_right[1, :])]
    sigma_k = inv(inv(sigma_carrot) + C' * inv(covariance_z) * C)
    mu_k = sigma_k * (inv(sigma_carrot) * mu_carrot + C' * inv(covariance_z) * zk)
    # println(mu_k)

    return mu_k, sigma_k
end
