using VehicleSim
using Test

function test_perception_h_jac()

    # variable set up for test
    n = 7
    epsilon = 1e-6

    # parameters for perception_jac_hx
    corner = [3.5 2.1 1.4]
    corner_id = 3
    x_other = [3 7 0.2 2 13 6 5] # [p1 p2 theta vel l w h]
    x_ego = [2 5 0.4 2 13.2 5.7 5.3]
    cam_id = 2

    J = perception_jac_hx(corner, corner_id, x_other, x_ego, cam_id)

    # perception_jac_hx goes from Rn to Rm, where n = and m =
    # We want to compute Jacobian of f at some point x
    # Let J = jac at that point
    for i = 1:n
        ei = zeros(n)
        ei[i] = epsilon
        df = (perception_jac_hx(corner, corner_id, x_other + ei, x_ego, cam_id)
              -
              perception_jac_hx(corner, corner_id, x_other, x_ego, cam_id)) / epsilon

        # df should equal approximately J[:, i]
        @test isapprox(df, J[:, i]; atol=1e-6)

    end
end