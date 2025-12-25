# Generalized Assignment Problem – Threshold Accepting

This repository contains a Python implementation of a **Threshold Accepting (TA) metaheuristic**
for the **Generalized Assignment Problem (GAP)**.

The algorithm assigns jobs to agents while respecting agent capacity constraints and minimizing
total assignment cost. Solution quality is evaluated against a **linear programming relaxation
lower bound** solved using **Gurobi**.

---

## Problem Description

Given:
- A set of agents with limited capacities
- A set of jobs to be assigned
- Assignment costs and resource consumptions

Each job must be assigned to exactly one agent, and the total resource consumption of each agent
must not exceed its capacity.

---

## Approach

- Threshold Accepting (TA) metaheuristic
- Random initial solution and neighborhood exploration
- Capacity violation handled via penalty mechanism
- Adaptive threshold decay
- LP relaxation lower bound computed with **Gurobi (gurobipy)**

---
```
## Repository Structure

├── main.py
├── data/
│ ├── problem1 instance.xlsx
│ ├── problem2 instance.xlsx
│ └── problem3 instance.xlsx
├── outputs/
│ ├── cost_vs_lower_bound.png
│ └── threshold_vs_iterations.png
└── README.md
```

---

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- **Gurobi Optimizer** (with a valid license)

> ⚠️ The lower bound computation requires Gurobi.  
> If Gurobi is not installed or licensed, the code will not run.

---

## How to Run

Clone the repository or download it as a ZIP file, then run:

```bash
python main.py

The selected instance file is specified directly in main.py:
file_path = "data/problem3 instance.xlsx"

Output:
- Iteration-wise progress (printed periodically)

- Final assignment solution and capacity usage

- Total cost and penalty

- Execution time

Result Plots:
- Cost vs. Lower Bound

- Threshold over Iterations

---
