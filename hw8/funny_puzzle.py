import heapq

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    from_coords = [(i // 3, i % 3) for i in range(len(from_state))]
    to_coords = [(i // 3, i % 3) for i in range(len(to_state))]

    total_distance = 0
    for i in range(len(from_state)):
        if from_state[i] != 0:
            from_row, from_col = from_coords[i]
            correct_position = to_state.index(from_state[i])
            to_row, to_col = to_coords[correct_position]
            total_distance += abs(from_row - to_row) + abs(from_col - to_col)

    return total_distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    succ_states = []

    empty_indices = [i for i, tile in enumerate(state) if tile == 0]
    empty_positions = [(idx // 3, idx % 3) for idx in empty_indices]

    moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    for empty_row, empty_col in empty_positions:
        for dr, dc in moves:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_state = state[:]
                new_index = new_row * 3 + new_col
                new_state[empty_row * 3 + empty_col], new_state[new_index] = new_state[new_index], new_state[empty_row * 3 + empty_col]
                if new_state != state:
                    succ_states.append(new_state)

    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    open_queue = []
    visited = set()
    heapq.heappush(open_queue, (get_manhattan_distance(state), state, 0, []))  # Initial state with f-score, state, g-score, path

    max_length = 1
    while open_queue:
        f_score, current_state, g_score, path = heapq.heappop(open_queue)
        visited.add(tuple(current_state))
        path.append(current_state)
        if current_state == goal_state:
            for idx, state in enumerate(path):
                print(state, "h={}".format(get_manhattan_distance(state)), "moves: {}".format(idx))
            print("Max queue length: {}".format(max_length))
            return

        successors = get_succ(current_state)
        for succ_state in successors:
            if tuple(succ_state) not in visited:
                succ_g_score = g_score + 1
                succ_f_score = succ_g_score + get_manhattan_distance(succ_state)
                max_length = max(max_length, len(open_queue) + 1)

                heapq.heappush(open_queue, (succ_f_score, succ_state, succ_g_score, path.copy()))

if __name__ == "__main__":

    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([2,5,1,4,0,6,7,0,3])
    print()

    solve([4,3,0,5,1,6,7,2,0])
    print()
