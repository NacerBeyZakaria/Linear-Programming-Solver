from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QComboBox, QLineEdit, QTextEdit, QPushButton, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import sys
import re
import numpy as np
from simplex_solver import simplex_linprog, educational_simplex
from graphical_solver import plot_graphical, feasible_vertices_2d, eval_objective_at_points

def parse_variables(objective_text, constraints_text):
    variable_names = set()
    var_regex = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)')
    variable_names.update(var_regex.findall(objective_text))
    for line in constraints_text.splitlines():
        variable_names.update(var_regex.findall(line))
    variable_names = sorted(variable_names, key=lambda x: [int(t) if t.isdigit() else t for t in re.split('(\d+)', x)])
    return {name: idx for idx, name in enumerate(variable_names)}, variable_names

def parse_expression(expr_text, var_map):
    expr_text = expr_text.replace(" ", "")
    coef_regex = re.compile(r'([+-]?\d*\.?\d*)?([a-zA-Z_][a-zA-Z_0-9]*)')
    coefs = [0.0] * len(var_map)
    for coef_str, var_name in coef_regex.findall(expr_text):
        if var_name in var_map:
            if coef_str in ("", "+", "-"):
                coef_val = float(coef_str + "1") if coef_str else 1.0
            else:
                coef_val = float(coef_str)
            coefs[var_map[var_name]] += coef_val
    return coefs

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Programming Solver")
        self.resize(850, 600)
        self.dark_mode = True

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.layout = QVBoxLayout(main_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_layout = QHBoxLayout()
        title_layout.addStretch(1)
        self.title_label = QLabel("Linear Programming Solver")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Arial", 60, QFont.Weight.Bold))
        self.title_label.setStyleSheet("""
            color: #00e1ff;
            padding: 24px;
            font-size: 60px;
            border: none;
            background: none;
        """)
        title_layout.addWidget(self.title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        title_layout.addStretch(1)
        self.btnToggleTheme = QPushButton("🌙 Toggle Theme")
        self.btnToggleTheme.setFixedWidth(180)
        self.btnToggleTheme.clicked.connect(self.toggle_theme)
        title_layout.addWidget(self.btnToggleTheme, alignment=Qt.AlignmentFlag.AlignRight)
        self.layout.addLayout(title_layout)

        self.layout.addWidget(QLabel("Objective Type:"))
        self.comboObjective = QComboBox()
        self.comboObjective.addItems(["Maximize", "Minimize"])
        self.layout.addWidget(self.comboObjective)

        self.layout.addWidget(QLabel("Objective Function (ex: 3x + 2y or 3profit + 2cost):"))
        self.inputObjective = QLineEdit()
        self.layout.addWidget(self.inputObjective)

        self.layout.addWidget(QLabel("Constraints (each line one constraint, ex: 2x + y <= 100 or x + y >= 10):"))
        self.inputConstraints = QTextEdit()
        self.inputConstraints.setPlaceholderText("Example:\n3x + 2y <= 18\nx <= 4\ny >= 1")
        self.layout.addWidget(self.inputConstraints)

        self.btnSolve = QPushButton("Solve Problem")
        self.btnSolve.setFixedHeight(40)
        self.btnSolve.clicked.connect(self.solve_problem)
        self.layout.addWidget(self.btnSolve)

        self.layout.addWidget(QLabel("Results:"))
        self.outputResult = QTextEdit()
        self.outputResult.setReadOnly(True)
        self.layout.addWidget(self.outputResult)

        self.apply_dark_theme()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.apply_light_theme()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #121212; color: #ffffff; font-family: 'Segoe UI'; font-size: 13px; }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #1e1e1e; border: 1px solid #555; color: #ffffff; padding: 5px;
            }
            QPushButton {
                background-color: #0078D7; color: white; border-radius: 6px; padding: 6px;
            }
            QPushButton:hover { background-color: #1495FF; }
        """)

    def apply_light_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #f4f4f4; color: #000000; font-family: 'Segoe UI'; font-size: 13px; }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff; border: 1px solid #aaa; color: #000000; padding: 5px;
            }
            QPushButton {
                background-color: #0078D7; color: white; border-radius: 6px; padding: 6px;
            }
            QPushButton:hover { background-color: #1495FF; }
        """)

    def solve_problem(self):
        try:
            objective_text = self.inputObjective.text().strip()
            constraints_text = self.inputConstraints.toPlainText().strip()
            objective_type = "max" if self.comboObjective.currentText() == "Maximize" else "min"
            var_map, var_names = parse_variables(objective_text, constraints_text)
            c_original = np.array(parse_expression(objective_text, var_map), dtype=float)
            A, b, senses = [], [], []
            bounds = [(0, None)] * len(var_map)
            constraint_regex = re.compile(r'(.+?)(<=|>=|=)(.+)')
            simple_tableau_only = True

            for line in constraints_text.splitlines():
                if not line.strip(): continue
                m = constraint_regex.match(line.replace(" ", ""))
                if not m: raise ValueError(f"Invalid constraint format: {line}")
                lhs, sense, rhs_str = m.groups()
                row = parse_expression(lhs, var_map)
                nonzero = [i for i, x in enumerate(row) if abs(x) > 1e-9]
                if len(nonzero) == 1 and sense in (">=", "=") and sum(abs(x) for x in row) == 1.0:
                    idx = nonzero[0]
                    val = float(rhs_str) / row[idx]
                    if sense == ">=":
                        bounds[idx] = (max(bounds[idx][0], val), bounds[idx][1])
                        simple_tableau_only = False
                    elif sense == "=":
                        bounds[idx] = (val, val)
                        simple_tableau_only = False
                    continue
                A.append(row)
                b.append(float(rhs_str))
                senses.append(sense)
                if sense != "<=": simple_tableau_only = False

            if len(A) == 0:
                self.outputResult.setPlainText("No valid constraints entered.")
                return

            if not simple_tableau_only:
                x, z, trace, status = simplex_linprog(c_original, A, b, senses, bounds, maximize=(objective_type=="max"))
                result_text = "⚡️Professional solver used for >=, bounds, or equality constraints.\n\n"
            else:
                x, z, trace, status = educational_simplex(c_original, A, b, maximize=(objective_type == "max"))
                result_text = "Simplex Trace Mode (<= only):\n\n"

            if x is None:
                self.outputResult.setPlainText(f"❌ Problem: {status}")
                return

            result_text += "✅ Optimal solution found:\n"
            for idx, val in enumerate(x):
                result_text += f"{var_names[idx]} = {val:.6f}\n"
            result_text += f"Z = {z:.6f}\n\n"
            if trace and simple_tableau_only:
                result_text += "--- Simplex tableau trace ---\n"
                for k, tab in enumerate(trace):
                    result_text += f"Iteration {k}:\n{tab}\n\n"
            self.outputResult.setPlainText(result_text)

            # --- ALWAYS SHOW GRAPHICAL FOR 2 VARS ---
            if len(var_names) == 2:
                constraints_std = []
                for row, rhs, sense in zip(A, b, senses):
                    arr = np.array(row)
                    if sense == "<=":    # as-is
                        constraints_std.append((arr, rhs))
                    elif sense == ">=":  # flip to <= form
                        constraints_std.append((-arr, -rhs))
                    elif sense == "=":   # split as <= and >=
                        constraints_std.append((arr, rhs))
                        constraints_std.append((-arr, -rhs))
                arrx = np.zeros(len(var_names)); arrx[0] = -1
                arry = np.zeros(len(var_names)); arry[1] = -1
                constraints_std.append((arrx, 0))
                constraints_std.append((arry, 0))
                try:
                    best_vertex, best_value = plot_graphical(c_original, constraints_std, objective_type=objective_type)
                    self.outputResult.append(
                        f"\nGraphical best point: ({var_names[0]},{var_names[1]}) = {best_vertex} with Z = {best_value:.6f}")
                except Exception as e:
                    self.outputResult.append(f"\nGraphical method error: {str(e)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
