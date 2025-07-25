import random

import numpy as np
from scipy.optimize import curve_fit, least_squares
from sympy import symbols, lambdify

from Symbolic import symbolic_evolution
import sympy as sp


def complex_fit(batch_x, batch_y, decay=None):
    import numpy as np
    import warnings
    from scipy.optimize import curve_fit, OptimizeWarning

    def model_func(t, a, w):
        x_model = np.cos(w * t) * np.exp(-a * t)
        y_model = -np.sin(w * t) * np.exp(-a * t)
        return np.concatenate([x_model, y_model])

    # Gather data from batch_x and batch_y.
    data_x = [batch_x.get_zi(i) for i in range(batch_x.n)]
    data_y = [batch_y.get_zi(i) for i in range(batch_x.n)]
    parameters = []

    for i in range(len(data_x)):
        t_points = batch_x.delay
        z_points = np.concatenate([np.array(data_x[i]), np.array(data_y[i])])

        # Check for empty data to avoid runtime warnings.
        if z_points.size == 0:
            if decay is not None:
                fitted_params = np.array([decay[i], 100])
            else:
                fitted_params = np.array([100, 100])
            parameters.append(fitted_params)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                if decay is not None:
                    # When decay is provided, fix 'a' and only fit for 'w'.
                    a_value = decay[i]
                    initial_guess = [1]  # Only fit for w.
                    bounds = ([0], [2 * np.pi])  # Bounds for w.
                    popt, _ = curve_fit(lambda t, w: model_func(t, a_value, w),
                                        t_points, z_points,
                                        p0=initial_guess, bounds=bounds)
                    fitted_params = np.array([a_value, popt[0]])
                else:
                    initial_guess = [1, 1]
                    bounds = ([-12, 0], [12, 2 * np.pi])
                    popt, _ = curve_fit(model_func, t_points, z_points,
                                        p0=initial_guess, bounds=bounds)
                    fitted_params = np.abs(popt)
        except Exception as e:
            # Fallback values if fitting fails.
            if decay is not None:
                fitted_params = np.array([decay[i], 100])
            else:
                fitted_params = np.array([100, 100])
        parameters.append(fitted_params)
    return parameters


def fit_X(batch_x, decay=None):
    import warnings
    from scipy.optimize import curve_fit, OptimizeWarning

    def model_func(t, a, w):
        return np.cos(w * t) * np.exp(-a * t)

    # Gather the data from batch_x.
    data_x = [batch_x.get_zi(i) for i in range(batch_x.n)]
    parameters = []

    for i in range(len(data_x)):
        t_points = batch_x.delay
        z_points = np.array(data_x[i])

        # Check if z_points is empty to avoid warnings from empty data.
        if z_points.size == 0:
            # Use fallback values if no data is present.
            if decay is not None:
                fitted_params = np.array([decay[i], 100])
            else:
                fitted_params = np.array([100, 100])
            parameters.append(fitted_params)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                if decay is not None:
                    # Fix a to the provided decay value, and only fit for w.
                    a_value = decay[i]
                    initial_guess = [0.5]  # Initial guess for w only.
                    bounds = ([0], [2 * np.pi])  # Bounds for w.
                    # Use a lambda to fix a_value.
                    popt, _ = curve_fit(lambda t, w: model_func(t, a_value, w),
                                        t_points, z_points,
                                        p0=initial_guess, bounds=bounds)
                    fitted_params = np.array([a_value, popt[0]])
                else:
                    # Fit both a and w.
                    initial_guess = [1, 0.5]
                    bounds = ([0, 0], [12, 2 * np.pi])
                    popt, _ = curve_fit(model_func, t_points, z_points,
                                        p0=initial_guess, bounds=bounds)
                    fitted_params = np.abs(popt)
        except Exception as e:
            # Fallback values in case fitting fails.
            if decay is not None:
                fitted_params = np.array([decay[i], 100])
            else:
                fitted_params = np.array([100, 100])
        parameters.append(fitted_params)
    return parameters


from scipy.optimize import minimize

# def fit_X(batch_x):
#     def model_func(t, a, w):
#         return (np.cos(w * t)) * np.exp(-a * t)
#
#     def absolute_error_loss(params, t_points, z_points):
#         a, w = params
#         predictions = model_func(t_points, a, w)
#         return np.sum(np.abs(predictions - z_points))
#
#     data_x = []
#     for i in range(batch_x.n):
#         data_x.append(batch_x.get_zi(i))
#
#     parameters = []
#     for i in range(len(data_x)):
#         t_points = batch_x.delay
#         z_points = np.concatenate([np.array(data_x[i])])
#
#         initial_guess = np.array([1, 1])  # Same initial guess as before
#         bounds = [(0, 6), (0, 2 * np.pi)]  # Same bounds as before
#
#         # Use minimize to optimize with absolute error
#         result = minimize(
#             absolute_error_loss,
#             initial_guess,
#             args=(t_points, z_points),
#             bounds=bounds,
#             method='L-BFGS-B'
#         )
#
#         if result.success:
#             params = result.x
#         else:
#             params = [100, 100]  # Fallback values in case of failure
#
#         parameters.append(np.abs(params))
#
#     return parameters
#

def one_by_one_fit(batch_x_detuning, batch_y_detuning, batch_x_crosstalk, batch_y_crosstalk):
    n = batch_x_detuning.n
    params = complex_fit(batch_x_detuning, batch_y_detuning)
    W = [params[i][1] for i in range(n)]
    decay = [params[i][0] for i in range(n)]

    crosstalk_qubits_measured = batch_x_crosstalk.qubits_measured
    if n > 1:
        params = complex_fit(batch_x_crosstalk, batch_y_crosstalk, decay=decay)
        J = [params[i][1] - W[crosstalk_qubits_measured[i]] for i in range(n - 1)]
    else:
        J = []
    return decay, W, J


def one_by_one_X(batch_x_detuning, batch_x_crosstalk):
    n = batch_x_detuning.n
    params = fit_X(batch_x_detuning)
    W = [params[i][1] for i in range(n)]
    decay = [params[i][0] for i in range(n)]

    if n > 1:
        crosstalk_qubits_measured = batch_x_crosstalk.qubits_measured
        params = fit_X(batch_x_crosstalk, decay=decay)
        J = [params[i][1] - W[crosstalk_qubits_measured[i]] for i in range(n - 1)]

        J = np.abs(J)
    else :
        J = []
    decay = np.abs(decay)
    W = np.abs(W)

    return decay, W, J


def full_complex_fit(batch_x, batch_y, neighbors=0, W_given=None, J_given=None, decay_given=None):
    n = batch_x.n

    data = []
    for i in range(len(batch_x.RamseyExperiments)):
        data.append(batch_x.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))
        data.append(batch_y.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))

    # symbolic_exp = symbolic_evolution.minimize_functions(n, times, neighbors=neighbors)

    symbolic_exp = symbolic_evolution.get_expectation_values_exp(n, neighbors=neighbors)
    t = symbols('t', real=True)
    w = symbols(f'ω0:{n}', real=True)
    j = symbols(f'j0:{n - 1}', real=True)
    a = symbols(f'a0:{n}', real=True)

    symbolic_exp = [lambdify([t, *w, *a, *j], expr, 'numpy') for expr in symbolic_exp]

    def model_func(t, *params):
        n = batch_x.n
        W = W_given
        J = J_given
        A = decay_given
        i = 0
        if decay_given is None:
            A = params[:n]
            i += n
        if W_given is None:
            W = params[i:i + n]
            i += n
        if J_given is None:
            J = params[i:i + n - 1]

        functions = np.array([expr(t, *W, *A, *J) for expr in symbolic_exp])
        functions = functions.T
        return np.concatenate(functions)
        # functions = np.array([symbolic_evolution.set_parameters(expr, W, J, A) for expr in symbolic_exp])
        # return functions

    # initial_guess = [1] * (2 * batch_x.n + (batch_x.n - 1))  # Adjusted to include J
    # initial_guess = [random.random() for i in range(2 * batch_x.n + (batch_x.n - 1))]
    initial_guess_A = []
    initial_guess_W = []
    initial_guess_J = []

    bounds_lower_W = []
    bounds_upper_W = []
    bounds_lower_J = []
    bounds_upper_J = []
    bounds_lower_A = []
    bounds_upper_A = []

    if decay_given is None:
        initial_guess_A = [random.random() for i in range(batch_x.n)]
        bounds_lower_A = [0] * batch_x.n
        bounds_upper_A = [4 * np.pi] * batch_x.n
    if W_given is None:
        initial_guess_W = [random.random() for i in range(batch_x.n)]
        bounds_lower_W = [-2 * np.pi] * batch_x.n
        bounds_upper_W = [2 * np.pi] * batch_x.n
    if J_given is None:
        initial_guess_J = [random.random() for i in range(batch_x.n - 1)]
        bounds_lower_J = [-2 * np.pi] * (batch_x.n - 1)
        bounds_upper_J = [2 * np.pi] * (batch_x.n - 1)
    initial_guess = np.concatenate([initial_guess_A, initial_guess_W, initial_guess_J])

    bounds_lower = np.concatenate([bounds_lower_A, bounds_lower_W, bounds_lower_J])
    bounds_upper = np.concatenate([bounds_upper_A, bounds_upper_W, bounds_upper_J])
    bounds = (bounds_lower, bounds_upper)

    # Perform the curve fitting
    t_points = batch_x.delay
    z_points = np.concatenate(data)
    params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess, bounds=bounds)
    guessed_decay = params[:batch_x.n][::-1]
    guessed_W = params[batch_x.n:2 * batch_x.n][::-1]
    guessed_J = params[2 * batch_x.n:3 * batch_x.n - 1][::-1]
    return guessed_decay, guessed_W, guessed_J


def full_complex_fit_modified(batch_x, batch_y, neighbors=0, W_given=None, J_given=None, decay_given=None):
    n = batch_x.n

    data = []
    for i in range(len(batch_x.RamseyExperiments)):
        data.append(batch_x.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))
        data.append(batch_y.RamseyExperiments[i].get_n_nearest_neighbors(neighbors))

    symbolic_exp = symbolic_evolution.get_expectation_values_exp(n, neighbors=neighbors)
    t = symbols('t', real=True)
    w = symbols(f'ω0:{n}', real=True)
    j = symbols(f'j0:{n - 1}', real=True)
    a = symbols(f'a0:{n}', real=True)

    symbolic_exp = [lambdify([t, *w, *a, *j], expr, 'numpy') for expr in symbolic_exp]

    def residuals(params, t, data):
        n = batch_x.n
        W = W_given if W_given is not None else params[:n]
        A = decay_given if decay_given is not None else params[n:2 * n]
        J = J_given if J_given is not None else params[2 * n:3 * n - 1]

        model_values = np.concatenate([expr(t, *W, *A, *J) for expr in symbolic_exp]).T
        return np.abs(data - model_values)

    initial_guess_A, initial_guess_W, initial_guess_J = [], [], []
    bounds_lower, bounds_upper = [], []

    # Adjust these bounds and initial guesses as per your model's needs
    if decay_given is None:
        initial_guess_A = [random.random() for _ in range(n)]
        bounds_lower.extend([0] * n)
        bounds_upper.extend([2 * np.pi] * n)
    if W_given is None:
        initial_guess_W = [random.random() for _ in range(n)]
        bounds_lower.extend([-2 * np.pi] * n)
        bounds_upper.extend([2 * np.pi] * n)
    if J_given is None:
        initial_guess_J = [random.random() for _ in range(n - 1)]
        bounds_lower.extend([-2 * np.pi] * (n - 1))
        bounds_upper.extend([2 * np.pi] * (n - 1))

    initial_guess = np.concatenate([initial_guess_A, initial_guess_W, initial_guess_J])
    bounds = (bounds_lower, bounds_upper)

    # Perform the optimization
    t_points = batch_x.delay
    z_points = np.concatenate(data)
    result = least_squares(residuals, initial_guess, bounds=bounds, args=(t_points, z_points), loss='soft_l1')

    params = result.x
    guessed_decay = params[:n]
    guessed_W = params[n:2 * n]
    guessed_J = params[2 * n:3 * n - 1]

    return guessed_decay, guessed_W, guessed_J


def percent_error(correct, fitted):
    MSE = []
    for param in range(len(correct)):
        mse = np.mean((np.array(correct[param]) - np.array(fitted[param]))** 2)
        MSE.append(mse)
    # mse = (correct - fitted) ** 2
    # relative_errors = mse
    return MSE


def calc_dist(fitted_values, correct_values):
    fitted_values = np.array(fitted_values)
    correct_values = np.array(correct_values)
    mse = (fitted_values - correct_values) ** 2 / len(fitted_values)
    precent_error = (np.sqrt(np.abs(mse)) / np.abs(correct_values)) * 100
    return precent_error


def full_complex_fit_test(batch_x_rot, batch_y_rot, batch_x_rot2, batch_y_rot2, neighbors=0, rot1=np.pi,
                          rot2=np.pi / 2):
    n = batch_x_rot.n

    data = []
    for i in range(len(batch_x_rot.RamseyExperiments)):
        data.append(batch_x_rot.RamseyExperiments[i].get_n_nearest_neighbors(0))
        data.append(batch_y_rot.RamseyExperiments[i].get_n_nearest_neighbors(0))

    # symbolic_exp = symbolic_evolution.minimize_functions(n, times, neighbors=neighbors)

    symbolic_exp = symbolic_evolution.get_expectation_values_exp(n, neighbors=0, rot=rot1)
    t = symbols('t', real=True)
    w = symbols(f'ω0:{n}', real=True)
    j = symbols(f'j0:{n - 1}', real=True)
    a = symbols(f'a0:{n}', real=True)

    threshold = 0.1
    symbolic_exp = [np.expand(function) for function in symbolic_exp]
    symbolic_exp = [func.as_ordered_terms() for func in symbolic_exp]

    filtered = []
    for func in symbolic_exp:
        filtered_terms = [term for term in func if abs(list(term.as_coefficients_dict().values())[0]) >= threshold]
        filtered_terms = sum(filtered_terms)
        filtered.append(filtered_terms)

    symbolic_exp = filtered

    symbolic_exp = [lambdify([t, *w, *a, *j], expr, 'numpy') for expr in symbolic_exp]

    def model_func(t, *params):
        n = batch_x_rot.n
        A = params[:n]
        W = params[n:2 * n]
        # J = params[2 * n:2 * n + n - 1]
        J = [0] * (n - 1)
        functions = np.array([expr(t, *W, *A, *J) for expr in symbolic_exp])
        functions = functions.T
        return np.concatenate(functions)
        # functions = np.array([symbolic_evolution.set_parameters(expr, W, J, A) for expr in symbolic_exp])
        # return functions

    # initial_guess = [random.random() for i in range(2 * batch_x.n + (batch_x.n - 1))]
    initial_guess = [random.random() for i in range(2 * batch_x_rot.n)]

    bounds_lower_A = [0] * batch_x_rot.n
    bounds_upper_A = [2 * np.pi] * batch_x_rot.n
    bounds_lower_W = [-2 * np.pi] * batch_x_rot.n
    bounds_upper_W = [2 * np.pi] * batch_x_rot.n

    bounds_lower = np.concatenate([bounds_lower_A, bounds_lower_W])
    bounds_upper = np.concatenate([bounds_upper_A, bounds_upper_W])

    bounds = (bounds_lower, bounds_upper)

    # Perform the curve fitting
    t_points = batch_x_rot.delay
    z_points = np.concatenate(data)
    params, params_covariance, *c = curve_fit(model_func, t_points, z_points, p0=initial_guess, bounds=bounds)
    guessed_decay = params[:batch_x_rot.n][::-1]
    guessed_W = params[batch_x_rot.n:2 * batch_x_rot.n][::-1]

    data2 = []
    for i in range(len(batch_x_rot.RamseyExperiments)):
        data2.append(batch_x_rot2.RamseyExperiments[i].get_n_nearest_neighbors(0))
        data2.append(batch_y_rot2.RamseyExperiments[i].get_n_nearest_neighbors(0))

    symbolic_exp = symbolic_evolution.get_expectation_values_exp(n, neighbors=0, rot=rot2)
    symbolic_exp = [np.expand(function) for function in symbolic_exp]
    symbolic_exp = [func.as_ordered_terms() for func in symbolic_exp]

    filtered = []
    for func in symbolic_exp:
        filtered_terms = [term for term in func if abs(list(term.as_coefficients_dict().values())[0]) >= threshold]
        filtered_terms = sum(filtered_terms)
        filtered.append(filtered_terms)

    symbolic_exp = filtered

    symbolic_exp = [lambdify([t, *w, *a, *j], expr, 'numpy') for expr in symbolic_exp]

    def model_func2(t, *params):
        n = batch_x_rot.n
        A = guessed_decay[::-1]
        W = guessed_W[::-1]
        J = params
        functions = np.array([expr(t, *W, *A, *J) for expr in symbolic_exp])
        functions = functions.T
        return np.concatenate(functions)
        # functions = np.array([symbolic_evolution.set_parameters(expr, W, J, A) for expr in symbolic_exp])
        # return functions

    initial_guess = [random.random() for i in range(batch_x_rot2.n - 1)]
    bounds_lower = [-2 * np.pi] * (batch_x_rot2.n - 1)
    bounds_upper = [2 * np.pi] * (batch_x_rot2.n - 1)
    bounds = (bounds_lower, bounds_upper)

    # Perform the curve fitting
    t_points = batch_x_rot2.delay
    z_points = np.concatenate(data2)
    params, params_covariance, *c = curve_fit(model_func2, t_points, z_points, p0=initial_guess, bounds=bounds)

    guessed_J = params[::-1]
    return guessed_decay, guessed_W, guessed_J


def mean_of_medians(errors_reshaped, k):
    mean_of_medians = []
    std_of_medians = []

    for errors in errors_reshaped:
        # Calculate medians for each reshaped array
        # medians = [np.median(np.array(errors[i])) for i in range(len(errors_reshaped))]

        # Split medians into k equal groups for bootstrapping
        group_size = len(errors) // k
        median_groups = [errors[i * group_size:(i + 1) * group_size] for i in range(k)]

        # Calculate the mean of medians for each group
        group_means = [np.sqrt((np.mean(group))) for group in median_groups]
        # group_means = [(np.mean(group)) for group in median_groups]

        # Compute the mean of these group means and its standard deviation
        m_m = np.mean(group_means)  # error calc returns errors squared
        s_m = np.std(group_means)/np.sqrt(group_size)
        mean_of_medians.append(m_m)
        std_of_medians.append(s_m)

    return mean_of_medians, std_of_medians


def mean_and_std(errors_reshaped):
    import numpy as np
    means = []
    stds = []
    for errors in errors_reshaped:
        # Filter out outliers: only keep errors <= 5
        filtered_errors = np.array(errors)[np.array(errors) <= 5]

        # If no data remains after filtering, assign NaN values.
        if len(filtered_errors) == 0:
            m = np.nan
            s_ratio = np.nan
        else:
            m = np.sqrt(np.mean(filtered_errors))
            s = np.std(filtered_errors) / np.sqrt(len(filtered_errors))
            s_ratio = s / m if m != 0 else np.nan
        means.append(m)
        stds.append(s_ratio)
    return means, stds
