import numpy as np
from scipy.optimize import linprog

def simplex_linprog(c, A, b, senses, bounds, maximize=True):
    c = np.array(c, dtype=float)
    n = len(c)
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for row, rhs, sense in zip(A, b, senses):
        row = np.array(row, dtype=float)
        if sense == "<=":
            A_ub.append(row)
            b_ub.append(rhs)
        elif sense == ">=":
            A_ub.append(-row)
            b_ub.append(-rhs)
        elif sense == "=":
            A_eq.append(row)
            b_eq.append(rhs)
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    c_solve = -c if maximize else c
    result = linprog(c_solve, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
    trace = []
    info = f"Linear Programming Solution:\nMethod: HiGHS\nIterations: {getattr(result, 'nit', 'N/A')}\nStatus: {result.message}\n"
    trace.append(info)
    if result.success:
        x = result.x
        z = float(np.dot(c, x))
        status = "optimal"
    else:
        x, z, status = None, None, result.message
    return x, z, trace, status

def educational_simplex(c, A, b, maximize=True, tol=1e-8, max_iters=100):
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c if maximize else c
    trace = [tableau.copy()]
    basis = [n + i for i in range(m)]
    for it in range(max_iters):
        reduced = tableau[-1, :-1]
        entering_candidates = np.where(reduced < -tol)[0] if maximize else np.where(reduced > tol)[0]
        if entering_candidates.size == 0:
            status = "optimal"
            break
        entering = entering_candidates[0]
        col = tableau[:m, entering]
        ratios = np.full(m, np.inf)
        for i in range(m):
            if col[i] > tol:
                ratios[i] = tableau[i, -1] / col[i]
        if np.all(np.isinf(ratios)):
            status = "unbounded"
            return None, None, trace, status
        leaving = int(np.argmin(ratios))
        pivot = tableau[leaving, entering]
        tableau[leaving, :] = tableau[leaving, :] / pivot
        for i in range(m+1):
            if i != leaving:
                tableau[i, :] -= tableau[i, entering] * tableau[leaving, :]
        basis[leaving] = entering
        trace.append(tableau.copy())
    else:
        status = f"max iterations ({max_iters}) reached"
    x_full = np.zeros(n + m)
    for row_idx in range(m):
        basic_col = basis[row_idx]
        if basic_col < n + m:
            x_full[basic_col] = tableau[row_idx, -1]
    x_result = x_full[:n]
    z_val = float(np.dot(c, x_result))
    return x_result, z_val, trace, status
