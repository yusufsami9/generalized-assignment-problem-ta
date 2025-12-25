import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB


start_time = time.time()

def load_excel_with_pandas(file_path):
    sheets = pd.read_excel(file_path, sheet_name=["Cost Matrix", "Resource Matrix", "Capacities"], header=None)
    allocation_costs = sheets["Cost Matrix"]
    resource_consumption = sheets["Resource Matrix"]
    resource_limits = sheets["Capacities"]
    return allocation_costs, resource_consumption, resource_limits

file_path = "data/problem3 instance.xlsx"
allocation_costs, resource_consumption, resource_limits = load_excel_with_pandas(file_path)
allocation_costs_array = allocation_costs.to_numpy()
resource_consumption_array = resource_consumption.to_numpy()
resource_limits_array = resource_limits.squeeze().to_numpy()
num_agents = allocation_costs_array.shape[0]
num_jobs = allocation_costs_array.shape[1]

def generate_random_solution(num_agents, num_jobs):
    solution = np.random.randint(0, num_agents, size=num_jobs)
    return solution

def calculate_cost(solution, allocation_costs, resource_consumption, resource_limits, iteration, max_iterations):
    agent_consumptions = np.zeros(num_agents)
    cost = 0
    for job, agent in enumerate(solution):
        agent_consumptions[agent] += resource_consumption[agent, job]
        cost += allocation_costs[agent, job]

    penalty_factor = np.exp(iteration / max_iterations)
    penalty = sum(max(0, agent_consumptions[agent] - resource_limits[agent]) * 1100 * penalty_factor for agent in range(num_agents))

    return cost, penalty

def generate_neighbors(solution, num_agents):
    neighbors = []
    for _ in range(50):
        new_solution = solution.copy()
        for _ in range(5):
            job_to_change = np.random.randint(len(solution))
            new_agent = np.random.randint(num_agents)
            new_solution[job_to_change] = new_agent
        neighbors.append(new_solution)
    return neighbors

def calculate_lower_bound(allocation_costs, resource_consumption, resource_limits):
    # Dimensions from data (avoid relying on globals)
    num_agents, num_jobs = allocation_costs.shape

    m = gp.Model("gap_lp_relaxation")
    m.setParam("OutputFlag", 0)  # no solver output

    # x[i,j] in [0,1] continuous
    x = m.addVars(num_agents, num_jobs, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: min sum c_ij x_ij
    m.setObjective(
        gp.quicksum(allocation_costs[i, j] * x[i, j] for i in range(num_agents) for j in range(num_jobs)),
        GRB.MINIMIZE
    )

    # Each job assigned exactly once: sum_i x_ij = 1
    for j in range(num_jobs):
        m.addConstr(gp.quicksum(x[i, j] for i in range(num_agents)) == 1, name=f"assign_{j}")

    # Capacity constraints: sum_j a_ij x_ij <= b_i
    for i in range(num_agents):
        m.addConstr(
            gp.quicksum(resource_consumption[i, j] * x[i, j] for j in range(num_jobs)) <= resource_limits[i],
            name=f"cap_{i}"
        )

    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Lower bound not optimal. Gurobi status code: {m.status}")

    return float(m.objVal)


def threshold_accepting(allocation_costs, resource_consumption, resource_limits, max_iterations, initial_threshold, threshold_decay):
    solution = generate_random_solution(num_agents, num_jobs)
    best_solution = solution.copy()
    best_cost, best_penalty = calculate_cost(solution, allocation_costs, resource_consumption, resource_limits, 0, max_iterations)

    threshold = initial_threshold

    costs_over_iterations = []
    thresholds_over_iterations = []

    for iteration in range(1, max_iterations + 1):
        neighbors = generate_neighbors(solution, num_agents)

        for neighbor in neighbors:
            neighbor_cost, neighbor_penalty = calculate_cost(neighbor, allocation_costs, resource_consumption, resource_limits, iteration, max_iterations)
            total_neighbor_cost = neighbor_cost + neighbor_penalty
            total_current_cost = best_cost + best_penalty

            if total_neighbor_cost <= total_current_cost + threshold:
                solution = neighbor
                best_cost = neighbor_cost
                best_penalty = neighbor_penalty
                best_solution = neighbor
                break

        threshold *= threshold_decay

        costs_over_iterations.append(best_cost)
        thresholds_over_iterations.append(threshold)

        is_feasible = "Feasible" if best_penalty == 0 else "Infeasible"
        if iteration % 1000 == 0 or iteration == 1 or iteration == max_iterations:
            is_feasible = "Feasible" if best_penalty == 0 else "Infeasible"
            print(
                f"Iteration {iteration}/{max_iterations} | Cost={best_cost:.0f} | Penalty={best_penalty:.2f} | {is_feasible} | Threshold={threshold:.4f}")

    agent_jobs = {agent + 1: [] for agent in range(num_agents)}
    for job, agent in enumerate(best_solution):
        agent_jobs[agent + 1].append(job + 1)

    agent_consumptions = [
        sum(resource_consumption[agent, j] for j in range(num_jobs) if best_solution[j] == agent)
        for agent in range(num_agents)
    ]

    return best_solution, best_cost, best_penalty, agent_jobs, agent_consumptions, costs_over_iterations, thresholds_over_iterations

def plot_cost_and_lower_bound(costs_over_iterations, lower_bound):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(costs_over_iterations) + 1), costs_over_iterations, marker='o', color='b', label='Cost')
    plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost and Lower Bound vs Iterations")
    plt.grid(alpha=0.6, linestyle='--')
    plt.legend()
    plt.show()

def plot_threshold_vs_iterations(thresholds_over_iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(thresholds_over_iterations) + 1), thresholds_over_iterations, marker='x', color='r', label='Threshold')
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold vs Iterations")
    plt.grid(alpha=0.6, linestyle='--')
    plt.legend()
    plt.show()

max_iterations = 10000
initial_threshold = 10000
threshold_decay = 0.999

lower_bound = calculate_lower_bound(allocation_costs_array, resource_consumption_array, resource_limits_array)
print(f"Lower Bound: {lower_bound}")

best_solution, best_cost, best_penalty, agent_jobs, agent_consumptions, costs_over_iterations, thresholds_over_iterations = threshold_accepting(
    allocation_costs_array, resource_consumption_array, resource_limits_array, max_iterations, initial_threshold, threshold_decay
)

print("\nThreshold Accepting Results:")
print(f"Best solution available (job-agent assignment): {[int(agent) + 1 for agent in best_solution]}")

print(f"Minimum cost: {best_cost}")
print(f"Total penalty: {best_penalty}")

print("\nAgents' jobs and resource consumptions:")
for agent, jobs in agent_jobs.items():
    print(f"Agent {agent}: Jobs {jobs}, Total Resource Usage: {agent_consumptions[agent - 1]} / {resource_limits_array[agent - 1]}")
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")
plot_cost_and_lower_bound(costs_over_iterations, lower_bound)
plot_threshold_vs_iterations(thresholds_over_iterations)
