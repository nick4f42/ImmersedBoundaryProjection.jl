struct StructuralMatrices
    M::Matrix{Float64}
    K::Matrix{Float64}
    Q::Matrix{Float64}
end

function StructuralMatrices(body::EulerBernoulliBeamBody)
    nf = 2 * npanels(body)
    M = zeros(nf, nf)
    K = zeros(nf, nf)
    Q = zeros(nf, nf)
    return StructuralMatrices(M, K, Q)
end

function update!(mats::StructuralMatrices, body::EulerBernoulliBeamBody, panels::PanelView)
    nb = npanels(body) # Number of body points
    nel = nb - 1 # Number of finite elements
    (; m, kb) = body # TODO: ke is not used here?
    (; M, K, Q) = mats

    # We will build these matrices by element and assemble in a loop
    for i_el in 1:nel
        Δx = panels.len[i_el]

        # Indices corresponding with the 4 unknowns associated w/ each element
        el_ind = @. (i_el - 1) * 2 + (1:4)

        # M_edup1 in the previous version
        Q_e =
            Δx / 420 * @SMatrix [
                156 22*Δx 54 -13*Δx
                22*Δx 4*Δx^2 -13*Δx -3*Δx^2
                54 13*Δx 156 -22*Δx
                13*Δx -3*Δx^2 -22*Δx 4*Δx^2
            ]

        M_e = m[i_el] * Q_e

        kb_el = kb[i_el]
        K_e =
            1 / (Δx^3) * @SMatrix [
                kb_el*12 kb_el*6*Δx -kb_el*12 kb_el*6*Δx
                kb_el*6*Δx kb_el*4*Δx^2 -kb_el*6*Δx kb_el*2*Δx^2
                -kb_el*12 -kb_el*6*Δx kb_el*12 -kb_el*6*Δx
                kb_el*6*Δx kb_el*2*Δx^2 -kb_el*6*Δx kb_el*4*Δx^2
            ]

        # Assemble into global matrices
        # Add contributions for each DOF in the element
        for (i, i_ind) in enumerate(el_ind), (j, j_ind) in enumerate(el_ind)
            M[i_ind, j_ind] += M_e[i, j]
            K[i_ind, j_ind] += K_e[i, j]
            Q[i_ind, j_ind] += Q_e[i, j]
        end
    end

    # Account for BCs (clamped-clamped)
    for bc::ClampIndexBC in body.bcs
        i = bc.i
        for j in 1:2
            k = 2 * (i - 1) + j
            M[k, :] .= 0.0
            M[:, k] .= 0.0
            K[k, :] .= 0.0
            K[:, k] .= 0.0
            Q[k, :] .= 0.0
            Q[:, k] .= 0.0 # TODO: this was commented out in old code?
            K[k, k] = 1.0
        end
    end

    return nothing
end
