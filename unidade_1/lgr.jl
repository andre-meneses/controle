cd(@__DIR__)
using Pkg
Pkg.activate(".")

using ControlSystems, Polynomials, Plots, Symbolics, PolynomialRoots
import PolynomialRoots

@variables s

struct System
    transferFn::TransferFunction{Continuous, ControlSystemsBase.SisoRational{Int64}}
    zeros::Vector{Complex{Float64}}
    poles::Vector{Complex{Float64}}
    np::Int
    nz::Int
end


function plot_rlocus(sys)
    rlocusplot(sys.transferFn)
    savefig("root_locus.pdf")
end


function System(G, H)
    GH = G*H

    zeros, poles, k = zpkdata(GH)
    zeros = zeros[1]
    poles = poles[1]

    nz = length(zeros)
    np = length(poles)

    System(GH, zeros, poles, np, nz)
end

function compute_sigma(sys)
    poles = sys.poles
    zeros = sys.zeros

    return (sum(poles) - sum(zeros))/(length(poles) - length(zeros))
end

function compute_phi(sys)
    np = sys.np
    nz = sys.nz

    q_max = np - nz - 1

    return [((2q + 1) * 180) / (np - nz) for q in 0:q_max]
end

function compute_derivative(sys)
    num = numpoly(sys.transferFn)[1].coeffs
    den = denpoly(sys.transferFn)[1].coeffs

    num_sym = sum([coeff * s^(i-1) for (i, coeff) in enumerate(num)])
    den_sym = sum([coeff * s^(i-1) for (i, coeff) in enumerate(den)])

    # Create the symbolic transfer function expression
    G_sym = -1*den_sym/num_sym

    # Differentiate the symbolic transfer function
    G_diff = Symbolics.Differential(s)(G_sym)
    G_diff = expand_derivatives(G_diff) # Simplify the differentiated expression

    return Symbolics.simplify(G_diff)
end

function compute_derivative_roots(sys)
    expr = Symbolics.value(compute_derivative(sys))
    numerator = arguments(expr)[1]
    coefficients = [coeff / s^(i-1) for (i, coeff) in enumerate(arguments(numerator))]

    poly = Polynomial(coefficients)
    poly = [Symbolics.value(i) for i in coeffs(poly)]

    return PolynomialRoots.roots(poly)

end

function main()
    G = tf([0,1], [1,6,8])
    H = tf([1,1], [1,4,0])

    sys = System(G,H)
    # println(compute_derivative_roots(sys))
    plot_rlocus(sys)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

