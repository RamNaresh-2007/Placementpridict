import heapq

def a_star(graph, heuristic, start, goal):
    # Priority queue: (f_cost, current_node)
    open_list = []
    heapq.heappush(open_list, (0, start))

    g_cost = {start: 0}
    parent = {start: None}

    while open_list:
        current_f, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1], g_cost[goal]

        for neighbor, cost in graph[current]:
            new_g = g_cost[current] + cost

            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                f_cost = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (f_cost, neighbor))
                parent[neighbor] = current

    return None, float("inf")


# Graph represented as adjacency list
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [('G', 5)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

# Heuristic values (estimated distance to goal G)
heuristic = {
    'A': 7,
    'B': 6,
    'C': 4,
    'D': 4,
    'E': 1,
    'F': 2,
    'G': 0
}

start = 'A'
goal = 'G'

path, cost = a_star(graph, heuristic, start, goal)

if path:
    print("Shortest Path:", " -> ".join(path))
    print("Total Cost:", cost)
else:
    print("No path found.")