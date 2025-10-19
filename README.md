# 🧮 Linear Program Solver

A **Windows application** for solving and visualizing **Linear Programming (LP)** problems using the **Simplex Method** and **Graphical Method**.

---

## 🎯 Purpose

This program helps users:

- ✅ Easily solve linear programming (LP) problems (maximize or minimize an objective function under constraints).  
- 📊 See both **algebraic (Simplex)** and **graphical** solutions for 2-variable problems.  
- 🧠 Learn and visualize LP feasible regions, optimal points, and Simplex tableau steps.

---

## ⚙️ How It Works

1. **Enter your objective function**  
   Example: `600x + 400y`

2. **Enter your constraints**, one per line  
   Example:  
   ```
   2x + y <= 2000
   4x + 8y <= 8000
   ```

3. **Select** whether you want to *maximize* or *minimize* your objective.

4. **Click** **Solve Problem**.

### The solver will:
- 🧮 Use **Simplex** or **linprog** to find the optimal solution.  
- 📜 Show a **step-by-step Simplex tableau** (for ≤ constraints).  
- 📈 Plot the **graphical solution** and all constraint boundaries (for 2-variable problems), highlighting the **optimal vertex**.

---

## 🧭 How To Use

1. **Download** and run `LinearProgramSolver.exe` from the **Releases** tab.  
2. Use the graphical interface:
   - Type the objective function in the first box.  
   - Enter each constraint on a new line in the constraints box.  
   - Choose **Maximize** or **Minimize** from the dropdown.  
   - Press **Solve Problem**.  
3. View the results:
   - Optimal solution variables and `Z` value.  
   - Simplex trace (if applicable).  
   - Interactive plot for 2-variable problems, showing constraint lines and the feasible region.

---

## 🖥️ Requirements

- **Windows 10/11**  
- **No installation needed** — just run the `.exe` file.

---

## 👩‍💻 About

Created for **students**, **researchers**, and **professionals** needing fast and visual LP solutions.
