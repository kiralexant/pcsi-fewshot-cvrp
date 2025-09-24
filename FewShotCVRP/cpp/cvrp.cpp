#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <limits>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iomanip>

namespace py = pybind11;

#ifdef DBG
#define DBG_PRINT_VECTOR(vec)                        \
    do                                               \
    {                                                \
        std::ostringstream oss;                      \
        oss << #vec << " = [";                       \
        for (size_t _i = 0; _i < (vec).size(); ++_i) \
        {                                            \
            oss << (vec)[_i];                        \
            if (_i + 1 < (vec).size())               \
                oss << ", ";                         \
        }                                            \
        oss << "]\n";                                \
        std::cout << oss.str();                      \
    } while (0)
#define DBG_PRINT_VAR(var)                                      \
    do                                                          \
    {                                                           \
        std::ostringstream oss;                                 \
        oss << #var << " = ";                                   \
        if constexpr (std::is_same_v<decltype(var), double>)    \
            oss << std::fixed << std::setprecision(5) << (var); \
        else                                                    \
            oss << (var);                                       \
        oss << "\n";                                            \
        std::cout << oss.str();                                 \
    } while (0)
#else
#define DBG_PRINT_VECTOR(vec) \
    do                        \
    {                         \
    } while (0)
#define DBG_PRINT_VAR(var) \
    do                     \
    {                      \
    } while (0)
#endif

class CVRP
{
public:
    const std::vector<double> depot_dist;        // size n
    const std::vector<std::vector<double>> dist; // shape n x n
    const std::vector<double> node_q;            // size n
    const int n;
    const double capacity;

    CVRP(std::vector<double> depot_dist_,
         std::vector<std::vector<double>> dist_,
         std::vector<double> node_q_,
         double capacity_)
        : depot_dist(std::move(depot_dist_)),
          dist(std::move(dist_)),
          node_q(std::move(node_q_)),
          n(static_cast<int>(node_q.size())),
          capacity(capacity_)
    {
        assert(static_cast<int>(depot_dist.size()) == n);
        assert(static_cast<int>(dist.size()) == n);
        for (const auto &row : dist)
        {
            assert(static_cast<int>(row.size()) == n);
        }
    }

    void check_is_permutation(const std::vector<int> &p) const
    {
        assert(static_cast<int>(p.size()) == n);
        std::vector<bool> used(n, false);
        for (int e : p)
        {
            assert(0 <= e && e < n);
            assert(!used[e]);
            used[e] = true;
        }
        for (bool v : used)
            assert(v);
    }

    // O(n)
    double greedy_fitness(const std::vector<int> &p, double max_interesting = -1.0) const
    {
        // check_is_permutation(p);

        int last_idx = -1;
        double result = 0.0;
        double curr_load = 0.0;

        for (int idx : p)
        {
            if (curr_load + node_q[idx] > capacity)
            {
                assert(last_idx >= 0 && "A single node exceeds vehicle capacity.");
                result += depot_dist[last_idx];
                last_idx = -1;
                curr_load = 0.0;
            }
            if (last_idx == -1)
                result += depot_dist[idx];
            else
                result += dist[last_idx][idx];

            if (max_interesting >= 0.0 && result > max_interesting)
                return result;

            curr_load += node_q[idx];
            last_idx = idx;
        }
        assert(last_idx >= 0);
        return result + depot_dist[last_idx];
    }

    // O(n^2)
    double dynamic_programming_fitness(const std::vector<int> &p, double max_interesting = -1.0) const
    {
        // check_is_permutation(p);

        // dp[i]: best cost if the last load ends at position i in p
        std::vector<double> dp(n, std::numeric_limits<double>::infinity());
        for (int i = 0; i < n; ++i)
        {
            double cap = 0.0, chain_sum = 0.0;

            for (int j = i; j >= 0; --j)
            {
                cap += node_q[p[j]];
                if (cap > capacity)
                    break;
                double cost = depot_dist[p[j]] + chain_sum + depot_dist[p[i]];

                if (j > 0)
                {
                    cost += dp[j - 1];
                    chain_sum += dist[p[j - 1]][p[j]];
                }
                dp[i] = std::min(dp[i], cost);

                DBG_PRINT_VECTOR(dp);
                DBG_PRINT_VAR(j);
                DBG_PRINT_VAR(i);
            }
        }
        return dp[n - 1];
    }

    // O(n^2)
    std::pair<double, std::vector<std::vector<int>>>
    dynamic_programming_fitness_and_ans(const std::vector<int> &p, double max_interesting = -1.0) const
    {
        // check_is_permutation(p);

        std::vector<double> dp(n, std::numeric_limits<double>::infinity());
        std::vector<int> prv(n, -1);
        for (int i = 0; i < n; ++i)
        {
            double cap = 0.0, chain_sum = 0.0;

            for (int j = i; j >= 0; --j)
            {
                cap += node_q[p[j]];
                if (cap > capacity)
                    break;
                double cost = depot_dist[p[j]] + chain_sum + depot_dist[p[i]];

                if (j > 0)
                {
                    cost += dp[j - 1];
                    chain_sum += dist[p[j - 1]][p[j]];
                }
                if (cost < dp[i])
                {
                    dp[i] = cost;
                    prv[i] = j;
                }
            }
        }
        std::vector<std::vector<int>> ans;
        int it = n - 1;
        while (it >= 0)
        {
            if (prv[it] < 0)
                return std::make_pair(dp[n - 1], std::vector<std::vector<int>>());
            std::vector<int> route;
            for (int i = prv[it]; i <= it; ++i)
                route.push_back(p[i]);
            ans.push_back(route);
            it = prv[it] - 1;
        }
        int it1 = 0, it2 = (int)ans.size() - 1;
        while (it1 < it2)
            std::swap(ans[it1++], ans[it2--]);
        return std::make_pair(dp[n - 1], ans);
    }
};

PYBIND11_MODULE(cvrp_cpp, m)
{
    m.doc() = "CVRP C++ implementation with pybind11 bindings";

    py::class_<CVRP>(m, "CVRP")
        // Constructor from Python lists (or any Python sequences)
        .def(py::init<std::vector<double>,
                      std::vector<std::vector<double>>,
                      std::vector<double>,
                      double>(),
             py::arg("depot_dist"),
             py::arg("dist"),
             py::arg("node_q"),
             py::arg("capacity"))

        // Methods
        .def("greedy_fitness", &CVRP::greedy_fitness,
             py::arg("p"), py::arg("max_interesting") = -1.0,
             "Greedy O(n) fitness")
        .def("dynamic_programming_fitness", &CVRP::dynamic_programming_fitness,
             py::arg("p"), py::arg("max_interesting") = -1.0,
             "DP O(n^2) fitness")
        .def("dynamic_programming_fitness_and_ans", &CVRP::dynamic_programming_fitness_and_ans,
             py::arg("p"), py::arg("max_interesting") = -1.0,
             "Return (cost, routes) from DP backtracking")

        // Fields (read-only in Python, mirroring dataclass(frozen=True))
        .def_readonly("depot_dist", &CVRP::depot_dist)
        .def_readonly("dist", &CVRP::dist)
        .def_readonly("node_q", &CVRP::node_q)
        .def_readonly("n", &CVRP::n)
        .def_readonly("capacity", &CVRP::capacity);
}
