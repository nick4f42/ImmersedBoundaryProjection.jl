module Solvers

using ...ImmersedBoundaryProjection
using ...ImmersedBoundaryProjection.Bodies
using ...ImmersedBoundaryProjection.Quantities
import ...ImmersedBoundaryProjection: advance!, solve!, statetype, solvertype
import ...ImmersedBoundaryProjection:
    timevalue, timeindex, timestep_scheme, conditions, discretized, gridstep
import ...ImmersedBoundaryProjection.Bodies:
    body_segment_length, bodypanels, prescribe_motion!

using EllipsisNotation
using FFTW
using IterativeSolvers
using LinearAlgebra
using LinearMaps
using StaticArrays
using FunctionWrappers: FunctionWrapper

export FreestreamFlow, PsiOmegaFluidGrid, UniformGrid, MultiLevelGrid
export gridstep, sublevel, baselevel

include("fluids.jl")

# Type of a problem where the bodies are all static relative to the discretization
const StaticBodyProblem{S} = Union{
    Problem{
        PsiOmegaFluidGrid{S,GlobalFrame},
        <:RigidBody{<:Union{GlobalFrame,DiscretizationFrame}},
    },
    Problem{<:PsiOmegaFluidGrid{S},RigidBody{DiscretizationFrame}},
}

include("states.jl")
include("fluid-ops.jl")
include("structure-ops.jl")
include("coupling.jl")
include("timestepping.jl")

end # module Solvers
