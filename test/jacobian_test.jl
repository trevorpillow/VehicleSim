using VehicleSim
using Test

# function test_perception_h_jac()
@testset "derivative tests" begin
    # variable set up for test
    n = 7
    epsilon = 1e-6

    # parameters for perception_h and perception_jac_hx
    x_other = [5.0 7.0 0.2 2.0 13.0 6.0 5.0] # [p1 p2 theta vel l w h]
    x_ego = [0.267 -0.534 -0.801 0.534 4.0 5.0 2.7] # [q1 q2 q3 q4 x y z] - ignored the other parts
    cam_id = 2

    # perepction_h measurement result
    zx, corner_ids, corners = VehicleSim.perception_h(x_other, x_ego, cam_id)
    # display("zx:")
    # display(zx)

    # Supposedly-correct Jacobian result
    Jx = VehicleSim.perception_jac_hx(corners[3], corner_ids[3], x_other, x_ego, cam_id)
    # display("Jacobian result:")
    # display(Jx)
    Jxn = similar(Jx)

    # perception_jac_hx goes from Rn to Rm, where n = and m =
    # We want to compute Jacobian of f at some point x
    # Let J = jac at that point
    for i = 1:n
        ei = [0.0 0.0 0.0 0.0 0.0 0.0 0.0] # same length as x_other
        ei[i] = epsilon
        zxi, corner_ids, corners = VehicleSim.perception_h(x_other + ei, x_ego, cam_id)
        df = (zxi - zx) / epsilon
        @test isapprox(df, J[:, i])
        # Jxn[:, i] = (zxi - zx) / epsilon
    end
    # @test isapprox(Jx, Jxn; atol=1e-6)
    # end
end