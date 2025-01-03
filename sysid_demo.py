import numpy as np
import casadi as cas
from casadi import DM, MX, Function
from sysid_utils import (make_n_step_simulator, make_gauss_newton_solver)


# In this example, we fit a nonlinear model to measurements
#
# This example uses more advanced constructs than the vdp* examples:
# Since the number of control intervals is potentially very large here, 
# we use memory-efficient Map and MapAccum, in combination with
# code generation.
#
# We will be working with a 2-norm objective:
# || y_measured - y_simulated ||_2^2
#
# This form is well-suited for the Gauss-Newton Hessian approximation.


############ SETTINGS #####################
N = 10000  # Number of samples
fs = 610.1  # Sampling frequency [hz]

param_truth = DM([5.625e-6, 2.3e-4, 1, 4.69])
param_guess = DM([5, 2, 1, 5])
scale = cas.vertcat(1e-6, 1e-4, 1, 1)


############ MODELING #####################
y  = MX.sym('y')
dy = MX.sym('dy')
u  = MX.sym('u')

states = cas.vertcat(y, dy)
controls = u

M = MX.sym("M")
c = MX.sym("c")
k = MX.sym("k")
k_NL = MX.sym("k_NL")

params = cas.vertcat(M, c, k, k_NL)

rhs = cas.vertcat(dy, (u - k_NL * y**3 - k*y - c*dy) / M)

# Form an ode function
ode = Function('ode', [states, controls, params], [rhs])


############ Creating a simulator ##########
n_steps_per_sample = 10
dt = 1 / fs / n_steps_per_sample
one_sample = make_n_step_simulator(ode, dt, states, controls, params, n_steps=n_steps_per_sample)


############ Simulating the system ##########
print("\nSimulating the system")

all_samples = one_sample.mapaccum("all_samples", N)

# Choose an excitation signal
np.random.seed(0)
u_data = DM(0.1 * np.random.random(N))

x0 = DM([0, 0])
X_measured = all_samples(x0, u_data, cas.repmat(param_truth, 1, N))

y_data = X_measured[0, :].T

# You may add some noise here
#y_data += 0.001 * np.random.random(N)
# When noise is absent, the fit will be perfect.


############ Identifying the simulated system: single shooting strategy ##########
print("\nIdentifying the simulated system: single shooting strategy")

# Note, it is in general a good idea to scale your decision variables such
# that they are in the order of ~0.1..100
X_symbolic = all_samples(x0, u_data, cas.repmat(params*scale, 1, N))

e = y_data - X_symbolic[0, :].T
nlp = {'x': params, 'f': 0.5 * cas.dot(e, e)}

solver = make_gauss_newton_solver(e, nlp, params)
sol = solver(x0=param_guess)

print(f"Solution: {sol['x'] * scale}")

assert(cas.norm_inf(sol["x"] * scale-param_truth) < 1e-8)


############ Identifying the simulated system: multiple shooting strategy ##########
print("\nIdentifying the simulated system: multiple shooting strategy")

# All states become decision variables
X = MX.sym("X", 2, N)

Xn = one_sample.map(N, 'openmp')(X, u_data.T, cas.repmat(params*scale, 1, N))

gaps = Xn[:, :-1] - X[:, 1:]

e = y_data-Xn[0, :].T

V = cas.veccat(params, X)

nlp = {'x': V, 'f': 0.5 * cas.dot(e, e), 'g': cas.vec(gaps)}

# Multiple shooting allows for careful initialization
yd = np.diff(y_data, axis=0) * fs
X_guess = cas.horzcat(y_data, cas.vertcat(yd, yd[-1])).T

x0 = cas.veccat(param_guess, X_guess)

solver = make_gauss_newton_solver(e, nlp, V)
sol = solver(x0=x0, lbg=0, ubg=0)

print(f"Solution: {sol['x'][:4] * scale}")

assert(cas.norm_inf(sol["x"][:4] * scale-param_truth) < 1e-8)