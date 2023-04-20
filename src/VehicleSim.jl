module VehicleSim

using ColorTypes
using Dates
using GeometryBasics
using MeshCat
using MeshCatMechanisms
using Random
using Rotations
using RigidBodyDynamics
using Infiltrator
using LinearAlgebra
using SparseArrays
using Suppressor
using Sockets
using Serialization
using StaticArrays
using DifferentialEquations
using DataStructures
using Ipopt
using Symbolics
using GLMakie #
using ProgressMeter #
using GLMakie.GeometryBasics #

include("view_car.jl")
include("objects.jl")
include("sim.jl")
include("client.jl")
include("control.jl")
include("sink.jl")
include("measurements.jl")
include("map.jl")
include("perception.jl")
include("example_project.jl")
include("path_finding.jl")
include("decision_making.jl")
include("trajectory.jl")

export server, shutdown!, keyboard_client, my_client

end
