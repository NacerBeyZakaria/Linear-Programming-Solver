import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def feasible_vertices_2d(constraints):
    vertices = []
    for (a1, b1), (a2, b2) in combinations(constraints, 2):
        A = np.vstack([a1, a2])
        if np.linalg.matrix_rank(A) < 2:
            continue
        try:
            pt = np.linalg.solve(A, np.array([b1, b2]))
            if all(np.dot(a, pt) <= b + 1e-6 for a, b in constraints):
                vertices.append(pt)
        except np.linalg.LinAlgError:
            continue
    # Add origin and intercepts if feasible
    origin = np.zeros(2)
    if all(np.dot(a, origin) <= b + 1e-6 for a, b in constraints):
        vertices.append(origin)
    for c in constraints:
        for i in range(2):
            if abs(c[0][i]) > 1e-8:
                pt = np.zeros(2)
                pt[i] = c[1]/c[0][i]
                if all(np.dot(a, pt) <= b + 1e-6 for a, b in constraints):
                    vertices.append(pt)
    # Remove duplicates
    out = []
    for v in vertices:
        if not any(np.allclose(v, vv, atol=1e-6) for vv in out):
            out.append(v)
    return np.array(out)

def eval_objective_at_points(obj_coeffs, points):
    return [float(np.dot(obj_coeffs, p)) for p in points]

def plot_graphical(obj_coeffs, constraints, objective_type="max", optimal_point=None):
    vertices = feasible_vertices_2d(constraints)
    if vertices.size == 0:
        raise ValueError("No feasible region found (no intersection points).")
    plt.figure(figsize=(7,6))
    xlim = [min(0, np.min(vertices[:,0])-100), np.max(vertices[:,0])+100]
    ylim = [min(0, np.min(vertices[:,1])-100), np.max(vertices[:,1])+100]
    x_range = np.linspace(xlim[0], xlim[1], 200)
    colors = ['magenta', 'green', 'orange', 'brown', 'purple', 'cyan']
    for idx, (a, b) in enumerate(constraints):
        if abs(a[1]) > 1e-10:
            y_line = (b - a[0] * x_range) / a[1]
            plt.plot(x_range, y_line, color=colors[idx%len(colors)], linestyle='--', label=f"Constraint {idx+1}: {a[0]:.2f}x + {a[1]:.2f}y = {b:.2f}")
        elif abs(a[0]) > 1e-10:
            x_val = b / a[0]
            plt.axvline(x=x_val, color=colors[idx%len(colors)], linestyle='--', label=f"Constraint {idx+1}: x = {x_val:.2f}")
    plt.scatter(vertices[:,0], vertices[:,1], color="blue", label="Feasible Vertices")
    for v in vertices:
        plt.annotate(f"({v[0]:.2f},{v[1]:.2f})", (v[0], v[1]), textcoords="offset points", xytext=(5,-10), fontsize=9)
    values = eval_objective_at_points(obj_coeffs, vertices)
    if objective_type.lower().startswith("max"):
        best_index = int(np.argmax(values))
    else:
        best_index = int(np.argmin(values))
    best_point = vertices[best_index]
    best_value = values[best_index]
    plt.scatter(best_point[0], best_point[1], color="red", s=100, label=f"Optimal ({best_point.round(2)})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Graphical Method - Feasible Region & Vertices")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.legend()
    plt.show()
    return best_point, best_value
