import xml.etree.ElementTree as ET
from math import sqrt
from random import randint, shuffle


# Distance between two points: Euclidean, round to nearest integer
def tsplib_dist(x1, y1, x2, y2):
    d2 = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
    return int(round(sqrt(d2)))


# The CVRP problem instance
class ProblemInstance:
    def __init__(self, xml_tree):
        instance_root = xml_tree.getroot()
        self.name = instance_root.find("info").find("name").text
        self.node_x = []
        self.node_y = []
        self.node_q = []

        # read nodes, of which the first is the depot
        for node in instance_root.find("network").find("nodes").findall("node"):
            node_id = int(node.get("id"))
            if node_id == 1:
                self.depot_x = int(node.find("cx").text)
                self.depot_y = int(node.find("cy").text)
            else:
                assert node_id == 2 + len(self.node_x)
                self.node_x.append(int(node.find("cx").text))
                self.node_y.append(int(node.find("cy").text))
                self.node_q.append(0)

        self.n = len(self.node_x)

        # read requests, one per node
        for request in instance_root.find("requests").findall("request"):
            trg = int(request.get("node")) - 2
            assert trg >= 0 and trg < self.n
            self.node_q[trg] = int(request.find("quantity").text)

        self.capacity = 0

        # read vehicle capacity
        for vp in instance_root.find("fleet").findall("vehicle_profile"):
            assert int(vp.find("departure_node").text) == 1
            assert int(vp.find("arrival_node").text) == 1
            assert self.capacity == 0
            self.capacity = int(vp.find("capacity").text)

        for q in self.node_q:
            assert q <= self.capacity  # otherwise we can't serve that node

        # cache all the distances
        self.depot_dist = [0] * self.n
        self.dist = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            self.depot_dist[i] = tsplib_dist(
                self.depot_x, self.depot_y, self.node_x[i], self.node_y[i]
            )
            for j in range(i):
                self.dist[i][j] = tsplib_dist(
                    self.node_x[i], self.node_y[i], self.node_x[j], self.node_y[j]
                )
                self.dist[j][i] = self.dist[i][j]

    def check_is_permutation(self, p):
        used = [False] * self.n
        for e in p:
            assert 0 <= e and e < self.n
            assert not used[e]
            used[e] = True
        for v in used:
            assert v

    # Greedy fitness evaluation, complexity O(n)
    def greedy_fitness(self, p, max_interesting=-1):
        # just assertions, not needed when the caller works
        self.check_is_permutation(p)

        last_idx = -1
        result = 0
        curr_load = 0

        # take nodes greedily in the order of p, while we can
        for idx in p:
            if curr_load + self.node_q[idx] > self.capacity:
                # if we cannot, drop the chain and start the new load
                result += self.depot_dist[last_idx]
                last_idx = -1
                curr_load = 0
            if last_idx == -1:
                result += self.depot_dist[idx]
            else:
                result += self.dist[last_idx][idx]

            # Speedup technique: if in the local search we are already worse than the parent,
            # return what we have to indicate that we are worse
            if max_interesting >= 0 and result > max_interesting:
                return result

            curr_load += self.node_q[idx]
            last_idx = idx

        return result + self.depot_dist[last_idx]

    # Dynamic programming fitness evaluation, complexity O(n^2)
    def dynamic_programming_fitness(self, p, max_interesting=-1):
        # just assertions, not needed when the caller works
        self.check_is_permutation(p)

        # dynamic programming: what's the best cost if the last load ended in node i?
        dp = [-1] * self.n
        for i in range(self.n):
            cap = self.node_q[p[i]]
            j = i
            chain_sum = 0
            # start the load with the current one assuming it's last,
            # scanning towards the head as long as we have the capacity
            while cap <= self.capacity:
                cost = self.depot_dist[p[j]] + chain_sum + self.depot_dist[p[i]]

                # Speedup technique: if the current partial cost is bigger than max_interesting,
                # anything which uses this node as the last in the load will be worse, so we can break
                if max_interesting >= 0 and cost > max_interesting:
                    break

                if j > 0:
                    cost += dp[j - 1]
                if dp[i] == -1 or dp[i] > cost:
                    dp[i] = cost
                if j == 0:
                    break
                j -= 1
                chain_sum += self.dist[p[j]][p[j + 1]]
                cap += self.node_q[p[j]]

        return dp[self.n - 1]


# Mutation: a 2-opt elementary mutation which chooses two random indices and
# reverses the permutation between them
def mutate_once_2opt(p):
    q = [e for e in p]
    i1 = randint(0, len(q) - 1)
    i2 = randint(0, len(q) - 1)
    if i1 > i2:
        i1, i2 = i2, i1
    while i1 < i2:
        q[i1], q[i2] = q[i2], q[i1]
        i1 += 1
        i2 -= 1
    return q


# Mutation: a swap-element mutation which chooses two random indices and
# exchanges the elements at these indices.
# Some people use 'swap' for swapping adjacent elements. English is ambiguous, so we don't do it.
def mutate_once_swap(p):
    q = [e for e in p]
    i1 = randint(0, len(q) - 1)
    i2 = randint(0, len(q) - 1)
    q[i1], q[i2] = q[i2], q[i1]
    return q


def main(argv):
    instance = ProblemInstance(ET.parse(argv[1]))

    print("name: " + instance.name)
    print("capacity: " + str(instance.capacity))
    print("nodes: " + str(instance.n))

    # random shuffle tests
    perm = [i for i in range(instance.n)]
    print("identity:")
    print("  greedy:  " + str(instance.greedy_fitness(perm)))
    print("  dynamic: " + str(instance.dynamic_programming_fitness(perm)))
    for t in range(10):
        shuffle(perm)
        print("shuffle " + str(t + 1) + ":")
        print("  greedy:  " + str(instance.greedy_fitness(perm)))
        print("  dynamic: " + str(instance.dynamic_programming_fitness(perm)))

    # infinite gradient descent
    best = perm
    # best_v = instance.dynamic_programming_fitness(best)
    best_v = instance.greedy_fitness(best)
    print("(1+1) init: " + str(best_v))
    iteration = 0
    while True:
        iteration += 1

        # the mutation is global here:
        # we invoke an elementary mutation for the number of times
        # distributed geometrically
        mut = mutate_once_2opt(best)
        while randint(0, 1) == 0:
            mut = mutate_once_2opt(mut)

        # mut_v = instance.dynamic_programming_fitness(mut, best_v)
        mut_v = instance.greedy_fitness(mut, best_v)
        if mut_v < best_v:
            print("  update to " + str(mut_v) + " on iteration " + str(iteration))
            best_v = mut_v
        if mut_v == best_v:
            best = mut


if __name__ == "__main__":
    import sys

    main(sys.argv)
