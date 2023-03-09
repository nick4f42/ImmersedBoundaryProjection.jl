module Bodies

using ..ImmersedBoundaryProjection
using ..ImmersedBoundaryProjection.Dynamics
using ..ImmersedBoundaryProjection.Curves
import ..ImmersedBoundaryProjection: _show

using StaticArrays

export AbstractBody, BodyGroup, Panels, PanelView, npanels, bodypanels, body_segment_length
export RigidBody, EulerBernoulliBeamBody, is_static
export EBBeamState, EBBeamStateView, ClampIndexBC, ClampParameterBC

# typeof(@view matrix[i:j, :])
const MatrixRowView{T} = SubArray{
    T,2,Matrix{T},Tuple{UnitRange{Int},Base.Slice{Base.OneTo{Int}}},false
}

# typeof(@view vector[i:j])
const VectorView{T} = SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}

"""
    AbstractBody

A structural body.
"""
abstract type AbstractBody end

Base.show(io::IO, ::MIME"text/plain", body::AbstractBody) = _show(io, body)

"""
    initial_pos!(xb::AbstractMatrix{Float64}, body::AbstractBody)

The initial `[x y]` panels on a body.
"""
function initial_pos! end

"""
    initial_lengths!(ds::AbstractVector{Float64}, body::AbstractBody)

The initial body segment lengths.
"""
function initial_lengths! end

"""
    npanels(body::AbstractBody)
    npanels(bodies::BodyGroup)
    npanels(panels::Panels)

The number of structural panels in a body or bodies.
"""
function npanels end

"""
    body_segment_length(fluid::AbstractFluid)

The optimal body segment length of a body simulated with `fluid`.
"""
function body_segment_length end

"""
    RigidBody(
        pos::AbstractMatrix{Float64},
        len::AbstractVector{Float64},
        frame::AbstractFrame=DiscretizationFrame()
    ) :: AbstractBody

A rigid body with optional prescribed motion.

# Arguments
- `pos`: Rows of `[x y]` points that define the body.
- `len`: Segment length at each point in `pos`.
- `motion`: Prescribed motion of the body.
"""
struct RigidBody{F<:AbstractFrame} <: AbstractBody
    pos::Matrix{Float64} # panel positions
    len::Vector{Float64} # panel lengths
    frame::F # prescribed motion
end

RigidBody(pos, len) = RigidBody(pos, len, DiscretizationFrame())
initial_pos!(xb, body::RigidBody) = xb .= body.pos
initial_lengths!(ds, body::RigidBody) = ds .= body.len

"""
    RigidBody(
        fluid::AbstractFluid,
        curve::Curve,
        frame::AbstractFrame=DiscretizationFrame(),
    ) :: AbstractBody

# Arguments
- `fluid`: The fluid that the body will be simulated in.
- `curve`: The curve that defines the shape of the rigid body.
- `frame`: Prescribed motion of the body.
"""
function RigidBody(fluid::AbstractFluid, curve::Curve, frame=DiscretizationFrame())
    segments = partition(curve, body_segment_length(fluid))
    return RigidBody(segments.points, segments.lengths, frame)
end

npanels(body::RigidBody) = length(body.len)

is_static(body::RigidBody{DiscretizationFrame}, ::AbstractFrame) = true
is_static(body::RigidBody{F}, ::F) where {F<:BaseFrame} = true
is_static(body::RigidBody, ::AbstractFrame) = false

function _show(io::IO, body::RigidBody, prefix)
    print(io, prefix, "RigidBody:")
    if get(io, :compact, false)
        print(io, " in frame ", body.frame)
    else
        ioc = IOContext(io, :limit => true, :compact => true)

        print(ioc, '\n', prefix, "  points = ")
        summary(ioc, body.pos)

        print(ioc, '\n', prefix, "   frame = ")
        summary(ioc, body.frame)
    end

    return nothing
end

"""
    ClampIndexBC(i::Int)

A boundary condition that index `i` on a body's points is clamped.
"""
struct ClampIndexBC
    i::Int
end

"""
    ClampParameterBC(t::Float64)

A boundary condition that `curve(t)` on a [`Curve`](@ref) is clamped.
"""
struct ClampParameterBC
    t::Float64
end

Base.@kwdef struct EulerBernoulliBeamBody <: AbstractBody
    xref::Matrix{Float64} # Reference locations about which displacements are determined
    x0::Matrix{Float64} # Initial locations
    ds0::Vector{Float64} # Line segment lengths on body in undeformed configuration
    kb::Vector{Float64} # Structural bending stiffness values
    ke::Vector{Float64} # Structural extensional stiffness values
    m::Vector{Float64} # Structural mass values
    bcs::Vector{ClampIndexBC} # Boundary conditions
end

initial_pos!(xb, body::EulerBernoulliBeamBody) = xb .= body.x0
initial_lengths!(ds, body::EulerBernoulliBeamBody) = ds .= body.ds0

"""
    search_closest(xs, x)

Return the index of the closest element in `xs` to `x` assuming `xs` is sorted.
"""
function search_closest(xs, x)
    i2 = searchsortedfirst(xs, x)

    i2 > lastindex(xs) && return lastindex(xs)
    i2 == firstindex(xs) && return firstindex(xs)

    i1 = i2 - 1
    return xs[i2] - x > x - xs[i1] ? i1 : i2
end

function EulerBernoulliBeamBody(
    fluid::AbstractFluid,
    curve::Curve,
    bcs::Vector{<:ClampParameterBC};
    kb::Float64,
    ke::Float64,
    m::Float64,
)
    segments = partition(curve, body_segment_length(fluid))
    x0 = xref = segments.points
    ds0 = segments.lengths

    nb = size(x0, 1)
    ke_vec = fill(ke, nb)
    kb_vec = fill(kb, nb - 1)
    m_vec = fill(m, nb - 1)

    s = cumsum(ds0)
    @. s = (s - s[1]) / (s[end] - s[1])
    bc_indices = map(bcs) do bc
        if !(0 <= bc.t <= 1)
            throw(
                DomainError(
                    "boundary condition parameter clamp must be between 0 and 1", bc.t
                ),
            )
        end
        ClampIndexBC(search_closest(s, bc.t))
    end

    return EulerBernoulliBeamBody(;
        xref, x0, ds0, kb=kb_vec, ke=ke_vec, m=m_vec, bcs=bc_indices
    )
end

npanels(body::EulerBernoulliBeamBody) = length(body.ds0)

function _show(io::IO, body::EulerBernoulliBeamBody, prefix)
    print(io, prefix, "EulerBernoulliBeamBody:")
    if get(io, :compact, false)
        print(io, " with bcs ", body.bcs)
    else
        ioc = IOContext(io, :limit => true, :compact => true)

        print(ioc, '\n', prefix, "  reference points = ")
        summary(ioc, body.xref)

        print(ioc, '\n', prefix, "    boundary conds = ", body.bcs)
    end

    return nothing
end

"""
    EBBeamStateView

A view into [`EBBeamState`](@ref).
"""
struct EBBeamStateView
    χ::MatrixRowView{Float64} # Structural displacements
    ζ::MatrixRowView{Float64} # Structural velocities
    ζdot::MatrixRowView{Float64} # Structural accels
end

"""
    EBBeamState

The state of deformation of all [`EulerBernoulliBeamBody`](@ref)s.
"""
struct EBBeamState
    χ::Matrix{Float64} # Structural displacements
    ζ::Matrix{Float64} # Structural velocities
    ζdot::Matrix{Float64} # Structural accels
    perbody::Vector{EBBeamStateView}
end

function EBBeamState(bodies::Vector{EulerBernoulliBeamBody})
    n = sum(npanels, bodies; init=0)

    χ = zeros(n, 2)
    ζ = zeros(n, 2)
    ζdot = zeros(n, 2)

    perbody = Vector{EBBeamStateView}(undef, length(bodies))
    i_panel = 0
    for (i, body) in enumerate(bodies)
        n_panel = npanels(body)
        r = i_panel .+ (1:n_panel)
        perbody[i] = @views EBBeamStateView(χ[r, :], ζ[r, :], ζdot[r, :])
        i_panel += n_panel
    end

    return EBBeamState(χ, ζ, ζdot, perbody)
end

"""
    BodyGroup(bodies::Vector{AbstractBody})

A collection of bodies.
"""
struct BodyGroup{B<:AbstractBody} <: AbstractVector{B}
    bodies::Vector{B} # all bodies
    deforming::Vector{Tuple{Int,EulerBernoulliBeamBody}} # (body_index, body)
    index_to_deform::Dict{Int,Int} # bodies[i] is deforming[index_to_deform[i]]
    npanel::Int
    function BodyGroup(bodies::Vector{B}) where {B<:AbstractBody}
        deforming = [
            (i, body) for (i, body) in enumerate(bodies) if body isa EulerBernoulliBeamBody
        ]
        index_to_deform = Dict(
            i_body => i_deform for (i_deform, (i_body, _)) in enumerate(deforming)
        )
        npanel = sum(npanels, bodies)
        return new{B}(bodies, deforming, index_to_deform, npanel)
    end
end

Base.size(bodies::BodyGroup) = size(bodies.bodies)
Base.getindex(bodies::BodyGroup, i) = bodies.bodies[i]
Base.IndexStyle(::BodyGroup) = IndexLinear()

npanels(bodies::BodyGroup) = bodies.npanel

Base.show(io::IO, ::MIME"text/plain", bodies::BodyGroup) = _show(io, bodies)

function _show(io::IO, bodies::BodyGroup, prefix)
    print(io, prefix)
    summary(io, bodies)
    print(io, ":\n")

    indent = prefix * "  "
    for body in bodies
        _show(io, body, indent)
        println(io)
    end
    return nothing
end

"""
    PanelView

View into sequences of panels in [`Panels`](@ref).
"""
struct PanelView
    pos::MatrixRowView{Float64}
    vel::MatrixRowView{Float64}
    len::VectorView{Float64}
    traction::MatrixRowView{Float64}
end

npanels(panels::PanelView) = length(panels.len)

"""
    Panels

A group of structural panels for multiple bodies.

Coordinates are given in the frame that the fluid is discretized in.
"""
struct Panels
    pos::Matrix{Float64} # panel positions
    vel::Matrix{Float64} # panel velocities
    len::Vector{Float64} # panel lengths
    traction::Matrix{Float64} # traction on each panel
    perbody::Vector{PanelView} # panels grouped by body
end

function Panels(bodies::BodyGroup)
    n = npanels(bodies)

    pos = zeros(n, 2)
    vel = zeros(n, 2)
    len = zeros(n)
    traction = zeros(n, 2)

    perbody = Vector{PanelView}(undef, length(bodies))
    i_panel = 0
    for (i, body) in enumerate(bodies)
        n_panel = npanels(body)
        r = i_panel .+ (1:n_panel)
        perbody[i] = @views PanelView(pos[r, :], vel[r, :], len[r], traction[r, :])
        i_panel += n_panel
    end

    for (body, panels) in zip(bodies, perbody)
        initial_pos!(panels.pos, body)
        initial_lengths!(panels.len, body)
    end

    return Panels(pos, vel, len, traction, perbody)
end

npanels(p::Panels) = length(p.len)

"""
    bodypanels(state::AbstractState) :: Panels

Return the [`Panels`](@ref) of a state.
"""
function bodypanels end

function prescribe_motion!(::PanelView, ::AbstractFrame, ::RigidBody, t::Float64)
    # TODO: Implement prescribed for other reference frames
    throw(ArgumentError("unsupported reference frame"))
end

function prescribe_motion!(
    ::PanelView, ::F1, ::RigidBody{F2}, ::Float64
) where {F1<:BaseFrame,F2<:Union{F1,DiscretizationFrame}}
    return nothing
end

function prescribe_motion!(
    panels::PanelView, ::F1, body::RigidBody{OffsetFrame{F2}}, t::Float64
) where {F1<:BaseFrame,F2<:Union{F1,DiscretizationFrame}}
    f = body.frame(t) # evaluate frame at time t
    r0 = f.r
    v0 = f.v
    Ω = f.Ω
    c = f.cθ # cos(θ)
    s = f.sθ # sin(θ)

    Rx = @SMatrix [c -s; s c]
    Rv = Ω * @SMatrix [-s -c; c -s]

    for (r, v, rb) in zip(eachrow.((panels.pos, panels.vel, body.pos))...)
        r .= r0 + Rx * rb
        v .= v0 + Rv * rb
    end

    return nothing
end

end # module Bodies
