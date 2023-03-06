module Quantities

using ..ImmersedBoundaryProjection
using ..ImmersedBoundaryProjection.Bodies
import ..ImmersedBoundaryProjection: timevalue

using HDF5

export Quantity, quantity
export GridQuantity, GridValue, GridValues, coordinates
export ArrayQuantity, ArrayValues
export MultiLevelGridQuantity, MultiLevelGridValue, MultiLevelGridValues

export flow_velocity, streamfunction, vorticity
export body_point_pos, body_point_vel, body_traction, body_lengths

"""
    Quantity

A function of [`AbstractState`](@ref).
"""
abstract type Quantity end

quantity(f::Quantity) = f
quantity(f) = ArrayQuantity(f)

include("quantity-types.jl")
include("quantity-funcs.jl")

end # module Quantities
