import warnings
import numpy as np
import casadi as cas
from casadi import DM, MX, Function


# Use just-in-time compilation if available
if cas.Importer.has_plugin('clang'):
    WITH_JIT = True
    COMPILER = 'clang'
elif cas.Importer.has_plugin('shell'):
    WITH_JIT = True
    COMPILER = 'shell'
else:
    warnings.warn(
        "Running without jit. This may result in very slow "
        "evaluation times"
    )
    WITH_JIT = False
    COMPILER = ''


def make_one_step_simulator(ode, dt, states, controls, params):
    """Create a one-step-ahead simulator function using Runge Kutta 4
    integration scheme.
    """

    # Runge Kutta 4 integration scheme
    k1 = ode(states, controls, params)
    k2 = ode(states + dt / 2.0 * k1, controls, params)
    k3 = ode(states + dt / 2.0 * k2, controls, params)
    k4 = ode(states + dt * k3, controls, params)
    states_final = states + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Create CasADi function
    return Function('one_step', [states, controls, params], [states_final])


def make_n_step_simulator(ode, dt, states, controls, params, n_steps=10):
    """Create a simulator function that simulates an n-step-ahead
    propagation of the system.
    """
    one_step = make_one_step_simulator(ode, dt, states, controls, params)
    X = states
    for _ in range(n_steps):
        X = one_step(X, controls, params)

    # Create CasADi function
    return Function('n_step', [states, controls, params], [X])


def make_gauss_newton_solver(e, nlp, V, with_jit=WITH_JIT, compiler=COMPILER):
    """Create a Gauss-Newton solver."""
    J = cas.jacobian(e, V)
    H = cas.triu(cas.mtimes(J.T, J))
    sigma = MX.sym("sigma")
    hess_lag = Function(
        'nlp_hess_l', 
        {'x': V, 'lam_f': sigma, 'hess_gamma_x_x': sigma * H},
        ['x', 'p', 'lam_f', 'lam_g'],
        ['hess_gamma_x_x'],
        {"jit": with_jit, "compiler": compiler}
    )
    return cas.nlpsol(
        "solver", 
        "ipopt", 
        nlp,
        {"hess_lag": hess_lag, "jit": with_jit, "compiler": compiler}
    )
