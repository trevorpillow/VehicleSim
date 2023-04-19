# The entire perception.jl file is working on ONE bounding box (i.e. only four corners of z is given).
# So, if there are two bounding boxes coming in per vehicle, the ekf function must be called twcie in
# example_project.jl's perception() function.

function perception_f(x, delta_t)
    # update xk = [p1 p2 theta vel l w h]
    # - updated-p1 = p1 + delta_time *cos(theta)*v
    # - updated-p2 = p2 + delta_time*sin(theta)*v
    theta_k = x[3]
    vel_k = x[4]
    x + delta_t * [vel_k * cos(theta_k), vel_k * sin(theta_k), 0, 0, 0, 0, 0]
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
    R = [cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]
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

    display("corners_body")
    display(corners_body)
    println()

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
    display("transform:")
    display(transform)
    println()

    # Section 2: Calculate the bounding boxes
    bbox = []
    # bbox_unrounded are used to calculate the jacobian of h in float to be more precise
    bbox_unrounded = []

    # NOTE: deal with having only 1 or 2 boxes here
    # similar to x_carrot = R * [q1 q2 q3] + t, turn points rotated cam frame
    corners_of_other_vehicle = zeros(Float64, 8, 3)
    for j = 1:num_corners
        corners_of_other_vehicle = [transform * [pt; 1] for pt in corners_body[j]]
    end
    display("corners_of_other_vehicle")
    display(corners_of_other_vehicle)
    println()

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

    # NOTE: DELETE THIS LATER
    # MANUALLY CHANGING THE CORNER[3] VALUES SO THAT I CAN CREATE JACOBIANS
    for j in corners_of_other_vehicle
        j[3] = 3.3655137916157862
    end

    # we are basically getting through each corner values in camera frame and 
    # keep updating the left, top, bottom, right values!
    for corner in corners_of_other_vehicle
        display("corner[3]")
        display(corner[3])
        println()

        # every point of corner in camera frame now
        if corner[3] < focal_len
            break
        end
        display("getting px and py")
        px = focal_len * corner[1] / corner[3]
        py = focal_len * corner[2] / corner[3]

        # update the corners
        if px < left
            left_cnr = corner
            corner_ids[2] = iter
        end
        if px > right
            right_cnr = corner
            corner_ids[4] = iter
        end
        if py < top
            top_cnr = corner
            corner_ids[1] = iter
        end
        if py > bot
            bot_cnr = corner
            corner_ids[3] = iter
        end

        left = min(left, px)
        right = max(right, px)
        top = min(top, py)
        bot = max(bot, py)

        iter = iter + 1
        display("moving on to next iter")
    end
    corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
    display("corners done")
    display(corners)
    println()

    # println(top)
    # println(bot)
    # println(left)
    # println(right)

    if top ≈ bot || left ≈ right || top > bot || left > right
        println("returning empty bbox")
        # out of frame - return empty bbox
        return bbox, corner_ids, corners
    else
        # update bbox
        display("now converting to pixels")
        top = convert_to_pixel(image_height, pixel_len, top)
        bot = convert_to_pixel(image_height, pixel_len, bot)
        left = convert_to_pixel(image_width, pixel_len, left)
        right = convert_to_pixel(image_width, pixel_len, right)
        push!(bbox, SVector(top, left, bot, right))
    end

    display("now returning perception_h")

    return bbox, corner_ids, corners
end


function calculate_J1_for_jac_hx(corner_id, x_other)
    # In perception_h, corners are set in the following format: 
    # corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
    l_mult = 0
    w_mult = 0
    h_mult = 0
    if corner_id == 1
        l_mult = 1
        w_mult = -1
        h_mult = -1
    elseif corner_id == 2
        l_mult = -1
        w_mult = -1
        h_mult = -1
    elseif corner_id == 3
        l_mult = 1
        w_mult = 1
        h_mult = -1
    elseif corner_id == 4
        l_mult = -1
        w_mult = 1
        h_mult = -1
    elseif corner_id == 5
        l_mult = 1
        w_mult = -1
        h_mult = 1
    elseif corner_id == 6
        l_mult = -1
        w_mult = -1
        h_mult = 1
    elseif corner_id == 7
        l_mult = 1
        w_mult = 1
        h_mult = 1
    else
        l_mult = -1
        w_mult = 1
        h_mult = 1
    end

    theta = x_other[3]
    l = 13.2
    w = 5.7
    h = 5.3

    [1 0 (0.5*(-sin(theta)*l_mult*l-cos(theta)*w_mult*w)) 0 (0.5*(cos(theta)*l_mult)) (0.5*(-sin(theta)*w_mult)) 0
        0 1 (0.5*(cos(theta)*l_mult*l+sin(theta)*w_mult*w)) 0 (0.5*sin(theta)*l_mult) (0.5*(-cos(theta)*w_mult)) 0
        0 0 0 0 0 0 (0.5+h_mult*0.5)]
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
    J3 = [1/corner[3] 0 -corner[1]/(corner[3])^2
        0 1/corner[3] -corner[2]/(corner[3])^2]

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
function perception_ekf(xego, delta_t, cam_id)
    # constant noise variables
    covariance_p = Diagonal([1^2, 1^2, 0.2^2, 0.4^2, 0.005^2, 0.003^2, 0.001^2])  # covariance for process model
    covariance_z = Diagonal([1^2, 1^2, 1^2, 1^2]) # covariance for measurement model
    num_steps = 25

    # initial states -- *change based on your state attributes
    x0 = [xego[1] + 2, xego[2] + 2, 0, xego[4], 8, 5, 5] # [p1 p2 theta vel l w h]
    # mean value of xk state of the OTHER car
    mu = zeros(7)
    sigma = Diagonal([1^2, 1^2, 0.2^2, 0.4^2, 0.005^2, 0.003^2, 0.001^2])

    # variables to keep updating
    # timesteps = []
    mus = [mu,] # the means
    sigmas = Matrix{Float64}[sigma,] # list of sigma_k's
    zs = Vector{Float64}[] # measurements of other car(s)

    x_prev = x0
    for k = 1:num_steps # for k = 1:something
        xk = perception_f(x_prev, delta_t)
        x_prev = xk
        zk, corner_ids, corners = perception_h(xk, xego, cam_id)

        # *All of the equations below are referenced from L17 pg.3 and HW4
        # Process model: P(xk | xk-1, bbxk) = N(A*x-1, sig_carrot))
        # - A = perception_jac_fx(x_prev, delta_t)
        # - sig_carrot = convariance_p + A * sigmas[k] * A'
        # - mu_carrot = perception_f(u[k], delta_t)
        A = perception_jac_fx(mus[k], delta_t)
        sig_carrot = covariance_p + A * sigmas[k] * A'
        mu_carrot = perception_f(mus[k], delta_t)

        # Measurement model
        # In perception_h, corners are set in the following format: 
        # corners = [top_cnr, left_cnr, bot_cnr, right_cnr]
        # *NOTE: might have to take only the top or bottom row depending the corner
        C_top = perception_jac_hx(corners[1], corner_ids[1], mu_carrot, xego, cam_id)
        C_left = perception_jac_hx(corners[2], corner_ids[2], mu_carrot, xego, cam_id)
        C_bot = perception_jac_hx(corners[3], corner_ids[3], mu_carrot, xego, cam_id)
        C_right = perception_jac_hx(corners[4], corner_ids[4], mu_carrot, xego, cam_id)
        # for debugging
        display("C_top, C_left, C_bot, C_right")
        display(C_top)
        display(C_left)
        display(C_bot)
        display(C_right)

        # now stack them up to have a 4 x 7 matrix
        # We only care about the top row for left and right
        # We only care about the bottom row for top and bottom
        C = [C_top[2, :]; C_left[1, :]; C_bot[2, :]; C_right[1, :]] # double check the order retrurned from h
        sigma_k = inv(inv(sig_carrot) + C' * inv(covariance_z) * C)
        mu_k = sigma_k * (inv(sigma_carrot) * mu_carrot + C' * inv(covariance_z) * zk)

        # update the variables
        push!(mus, mu_k)
        push!(sigmas, sigma_k)
        push!(zs, zk)
        # push!(gt_states, xk)
        # push!(timesteps, delta_t)
    end
end
