import warnings
import casadi as cas
from casadi import MX, Function


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


def make_one_step_simulator(ode, dt, states, inputs, params, name='RK4'):
    """Create a one-step-ahead simulator function using Runge Kutta 4
    integration scheme.

    Arguments:
        ode : Function 
            CasADi function for the righthand side of the ODE equation
            with the signature: ode(states, inputs, params).
        dt : float
            Timestep size.
        states : MX.sym
            Vector representing the state variables.
        inputs : MX.sym
            Vector representing the system inputs.
        params : MX.sym
            Vector of parameter values.
        name : str
            Name to give to returned function.

    Returns:
        Function(name, [states, inputs, params], [states_final])

    """

    # Runge Kutta 4 integration scheme
    k1 = ode(states, inputs, params)
    k2 = ode(states + dt / 2.0 * k1, inputs, params)
    k3 = ode(states + dt / 2.0 * k2, inputs, params)
    k4 = ode(states + dt * k3, inputs, params)
    states_final = states + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Create CasADi function
    return Function(
        name, 
        [states, inputs, params], 
        [states_final], 
        ['states', 'inputs', 'params'], 
        ['states_final']
    )


def make_n_step_simulator(ode, dt, states, inputs, params, n_steps=10, name=None):
    """Create a simulator function that simulates an n-step-ahead
    propagation of the system.

    Arguments:
        ode : Function 
            CasADi function for the righthand side of the ODE equation
            with the signature: ode(states, inputs, params).
        dt : float
            Timestep size.
        states : MX.sym
            Vector representing the state variables.
        inputs : MX.sym
            Vector representing the system inputs.
        params : MX.sym
            Vector of parameter values.
        n_steps : int
            Number of consequtive timesteps over which to repeat the 
            integration.
        name : str
            Name to give to returned function.  If name is None, a
            name will be automatically generated based on the name of the 
            ode function and n_steps.
    
    Returns:
        Function(name, [states, inputs, params], [X])

    """
    step = make_one_step_simulator(ode, dt, states, inputs, params)
    X = states
    for _ in range(n_steps):
        X = step(X, inputs, params)

    if name is None:
        name = f"{step.name()}_{n_steps}_steps"

    # Create CasADi function
    return Function(
        name, 
        [states, inputs, params], 
        [X], 
        ['states', 'inputs', 'params'], 
        ['X']
    )


def make_gauss_newton_solver(
    errors, 
    x, 
    g=None, 
    name="solver", 
    solver="ipopt", 
    with_jit=WITH_JIT, 
    compiler=COMPILER
):
    """Create a Gauss-Newton solver to solve an NLP of the form:

        minimize f(x)
        subject to:
            x_lb ≤ x ≤ x_ub
            g_lb ≤ g(x) ≤ g_ub

    using a sequential quadratic programming (SQP) approach.

    Starting from a given initial guess for the primal and dual variables 
    (x(0), λ(0)), SQP solves the NLP by iteratively computing local convex 
    quadratic approximations at the current iterate (x(k), λ(k)) and 
    solving them using a quadratic programming (QP) solver.

    For NLPs with a least-squares objective function f(x) = norm(R(x))^2, 
    it is often a good idea to use the so-called Gauss-Newton method, 
    which uses an approximation of the Hessian of the Lagrangian.

    For details, see:
      - https://www.syscop.de/files/2015ss/events/tempo/e3_gn.pdf

    """

    J = cas.jacobian(errors, x)
    H = cas.triu(cas.mtimes(J.T, J))
    sigma = MX.sym("sigma")
    options = {"jit": with_jit, "compiler": compiler}
    hess_lag = Function(
        'nlp_hess_lag',
        {'x': x, 'lam_f': sigma, 'hess_gamma_x_x': sigma * H},
        ['x', 'p', 'lam_f', 'lam_g'],
        ['hess_gamma_x_x'],
        options
    )
    nlp = {'x': x, 'f': 0.5 * cas.dot(errors, errors)}
    if g is not None:
        nlp['g'] = g
    options = {"hess_lag": hess_lag, "jit": with_jit, "compiler": compiler}

    return cas.nlpsol(name, solver, nlp, options)
