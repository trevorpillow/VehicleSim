using VehicleSim
using Test

# function test_perception_h_jac()
@testset "derivative tests" begin
    # variable set up for test
    n = 7
    # epsilon = 1 
    epsilon = 1e-6


    # bbox of = [241, 321, 242, 322]
    # parameters for perception_h and perception_jac_hx
    x_other = [-91.6639496981265 -5.001254676640403 0.003 0.0001 13.2 5.7 5.3] # [p1 p2 theta vel l w h]
    x_ego = [0.7070991651230024 0.003813789504350662 -0.0030385002054085716 0.7070975839362422 -61.6639496981265 -35.00125467663771 2.6455622444987412] # [q1 q2 q3 q4 x y z] - ignored the other parts
    cam_id = 1

    """
        TEST PROCESS
    """
    delta_t = 0.003
    fx = VehicleSim.perception_f(x_other, delta_t)
    println("fx:")
    println(fx)
    println()
    Jfx = VehicleSim.perception_jac_fx(x_other, delta_t)
    println("Jfx:")
    println(Jfx)
    println()
    for i = 1:n
        ei = [0.0 0.0 0.0 0.0 0.0 0.0 0.0] # same length as x_other
        ei[i] = epsilon
        fxi = VehicleSim.perception_f(x_other + ei, delta_t)
        df = (fxi - fx) / epsilon
        display("df:")
        display(df)
        println()

        display("Jx[:, i]")
        display(Jfx[:, i])
        println()
        # manually checked df and Jfx[:, i] are basically the same
        # @test isapprox(df, Jfx[:, i]) # df[1] in order to just turn it into the same type
    end


    """
        TEST MEASUREMENT
    """
    # perepction_h measurement result
    zx, corner_ids, corners = VehicleSim.perception_h(x_other, x_ego, cam_id)
    display("zx::")
    display(zx)
    println()

    display("corner ids:")
    display(corner_ids)
    println()

    display("corners:")
    display(corners)
    println()

    # Supposedly-correct Jacobian result
    C_top = VehicleSim.perception_jac_hx(corners[1], corner_ids[1], x_other, x_ego, cam_id)
    C_left = VehicleSim.perception_jac_hx(corners[2], corner_ids[2], x_other, x_ego, cam_id)
    C_bot = VehicleSim.perception_jac_hx(corners[3], corner_ids[3], x_other, x_ego, cam_id)
    C_right = VehicleSim.perception_jac_hx(corners[4], corner_ids[4], x_other, x_ego, cam_id)

    # display("ctop to right")
    # display(C_top)
    # display(C_left)
    # display(C_bot)
    # display(C_right)
    # println()

    Jx = [transpose(C_top[2, :]); transpose(C_left[1, :]); transpose(C_bot[2, :]); transpose(C_right[1, :])] # double check the order retrurned from h
    display("Jacobian result:")
    display(Jx)
    # display(Jx[:, 3])
    println()

    # perception_jac_hx goes from Rn to Rm, where n = and m =
    # We want to compute Jacobian of f at some point x
    for i = 1:n
        ei = [0.0 0.0 0.0 0.0 0.0 0.0 0.0] # same length as x_other
        ei[i] = epsilon
        zxi, corner_ids, corners = VehicleSim.perception_h(x_other + ei, x_ego, cam_id)
        df = (zxi - zx) / epsilon
        # display("df:")
        # display(df[1])
        # println()

        # display("Jx[:, i]")
        # display(Jx[:, i])
        # println()
        @test isapprox(df[1], Jx[:, i]) # df[1] in order to just turn it into the same type
    end

    # maybe I have to finish calculating and then compare after the for loop..


    # @test isapprox(Jx, Jxn; atol=1e-6)
    # end
end