import numpy as np
from cvxopt import matrix, solvers


def manhattan(p, q):
    return sum([abs(a - b) for a, b in zip(p, q)])


stations = [
    [0, 1], [1, 3], [2, 2], 
    [2, 1], [3, 0]
]


cost = [
    150, 150, 350,
    200, 100
]


def main():
    points = np.empty((0, 5), int)
    
    for y in range(0, 4):
        for x in range(0, 4):
            if [x, y] not in [[0, 2], [1, 2], [3, 2], [3, 3]]:
                point = list()
                for station in enumerate(stations):
                    if ((manhattan(station[1], [x, y]) > 3) or 
                        ([x, y] in [[1, 0], [0, 1], [1, 1]] and station[0] == 1) or
                        ([x, y] in [[0, 3], [1, 3]] and station[0] == 0)):
                        point.append(0)
                    else:
                        point.append(1)
                points = np.append(points, [point], axis=0)

    c = matrix(cost, tc='d')
    G = matrix(points * -1, tc='d')
    h = matrix([-1 for _ in range(len(points))], tc='d')
    res = solvers.lp(c, G, h, solver='glpk')

    print(f"status: {res['status']}")
    print(f"sum: {res['primal objective']}")
    print(f"stations: {list(np.array(res['x'], dtype=int))}")  


if __name__ == "__main__":
    main()
    