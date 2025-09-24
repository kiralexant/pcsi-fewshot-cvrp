#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "external/HillVallEA/HillVallEA/hillvallea.hpp"
#include "external/HillVallEA/HillVallEA/fitness.h"

namespace py = pybind11;

struct HillVallEAConfiguration
{
  const py::function f;
  const int number_of_parameters;
  const int maximum_number_of_evaluations;
  const int maximum_number_of_seconds;
  const std::vector<double> lower;
  const std::vector<double> upper;
  int random_seed;
  const std::string internal_optimizer_name;
  const std::string problem_name;
  double value_to_reach; // if the optimum is known, you can terminate HillVallEA if it found a solution with fitness below the value_to_reach (vtr)
  bool use_value_to_reach;
  double TargetTolFun; // acceptable quality gap between the global and local optima

  HillVallEAConfiguration(
      py::function f_,
      int number_of_parameters_,
      int maximum_number_of_evaluations_,
      int maximum_number_of_seconds_,
      std::vector<double> lower_,
      std::vector<double> upper_,
      int random_seed_,
      std::string internal_optimizer_name_ = "AMaLGaM-Univariate", // suggested by the authers
      std::string problem_name_ = "DefaultProblemName",
      double value_to_reach_ = 0.0,
      bool use_value_to_reach_ = false,
      double TargetTolFun_ = 1e-5 // default value
      )
      : f(std::move(f_)),
        number_of_parameters(number_of_parameters_),
        maximum_number_of_evaluations(maximum_number_of_evaluations_),
        maximum_number_of_seconds(maximum_number_of_seconds_),
        lower(std::move(lower_)),
        upper(std::move(upper_)),
        random_seed(random_seed_),
        internal_optimizer_name(std::move(internal_optimizer_name_)),
        problem_name(std::move(problem_name_)),
        value_to_reach(value_to_reach_),
        use_value_to_reach(use_value_to_reach_),
        TargetTolFun(TargetTolFun_)
  {
  }

  int internal_optimizer_name_to_num() const
  {
    // 0 = AMaLGaM, 1 = AMaLGaM-Univariate, 10 = CMSA-ES,  20 = iAMaLGaM, 21 = iAMaLGaM-Univariate
    // AMaLGaM-Univariate (1) is suggested
    if (internal_optimizer_name == "AMaLGaM")
      return 0;
    if (internal_optimizer_name == "AMaLGaM-Univariate")
      return 1;
    if (internal_optimizer_name == "CMSA-ES")
      return 10;
    if (internal_optimizer_name == "iAMaLGaM")
      return 20;
    if (internal_optimizer_name == "iAMaLGaM-Univariate")
      return 21;
    throw std::runtime_error("Error: Algorithm name " + internal_optimizer_name + " is not known.");
  }
};

namespace hillvallea
{
  class ObjectiveFunction : public fitness_t
  {
    HillVallEAConfiguration myconfig;

  public:
    ObjectiveFunction(const HillVallEAConfiguration &config) : myconfig(config)
    {
      number_of_parameters = config.number_of_parameters;
      maximum_number_of_evaluations = config.maximum_number_of_evaluations;
    }
    ~ObjectiveFunction() {}

    void get_param_bounds(vec_t &lower_, vec_t &upper_) const
    {
      lower_.resize(number_of_parameters);
      upper_.resize(number_of_parameters);
      for (int i = 0; i < number_of_parameters; ++i)
      {
        lower_[i] = myconfig.lower[i];
        upper_[i] = myconfig.upper[i];
      }
    }

    void define_problem_evaluation(hillvallea::solution_t &sol)
    {
      py::array_t<double> arr_x(sol.param.size(), sol.param.data());
      double fx = myconfig.f(arr_x).cast<double>();
      sol.f = fx;
      sol.penalty = 0.0;
    }

    std::string name() const { return myconfig.problem_name; }
  };
}

std::vector<std::vector<double>> optimize(const HillVallEAConfiguration &config)
{
  hillvallea::fitness_pt fitness_function = std::make_shared<hillvallea::ObjectiveFunction>(config);
  hillvallea::vec_t lower_range_bounds, upper_range_bounds;
  fitness_function->get_param_bounds(lower_range_bounds, upper_range_bounds);
  size_t local_optimizer_index = config.internal_optimizer_name_to_num();
  int maximum_number_of_evaluations = config.maximum_number_of_evaluations;
  int maximum_number_of_seconds = config.maximum_number_of_seconds;
  double value_to_reach = config.value_to_reach;
  bool use_vtr = config.use_value_to_reach;
  int random_seed = config.random_seed;

  // Output to test files
  bool write_generational_solutions = false;
  bool write_generational_statistics = false;
  std::string write_directory = "./";
  std::string file_appendix = ""; // can be used when multiple runs are outputted in the same directory

  //-----------------------------------------
  hillvallea::hillvallea_t opt(
      fitness_function,
      (int)local_optimizer_index,
      (int)fitness_function->number_of_parameters,
      lower_range_bounds,
      upper_range_bounds,
      lower_range_bounds,
      upper_range_bounds,
      maximum_number_of_evaluations,
      maximum_number_of_seconds,
      value_to_reach,
      use_vtr,
      random_seed,
      write_generational_solutions,
      write_generational_statistics,
      write_directory,
      file_appendix);
  opt.TargetTolFun = config.TargetTolFun;

  std::cout << "Running HillVallEA on the defined objective function" << std::endl;

  opt.run();

  std::vector<std::vector<double>> ans(opt.elitist_archive.size());
  for (size_t i = 0; i < opt.elitist_archive.size(); ++i)
    ans[i] = opt.elitist_archive[i]->param;
  return ans;
}

PYBIND11_MODULE(hillvallimpl, m)
{
  m.doc() = "HillVallEA Python bindings";

  py::class_<HillVallEAConfiguration>(m, "HillVallEAConfiguration")
      .def(py::init<py::function, int, int, int,
                    std::vector<double>, std::vector<double>, int,
                    std::string, std::string, double, bool, double>(),
           py::arg("f"),
           py::arg("number_of_parameters"),
           py::arg("maximum_number_of_evaluations"),
           py::arg("maximum_number_of_seconds"),
           py::arg("lower"),
           py::arg("upper"),
           py::arg("random_seed"),
           py::arg("internal_optimizer_name") = "AMaLGaM-Univariate",
           py::arg("problem_name") = "DefaultProblemName",
           py::arg("value_to_reach") = 0.0,
           py::arg("use_value_to_reach") = false,
           py::arg("TargetTolFun") = 1e-5)
      .def_readonly("number_of_parameters", &HillVallEAConfiguration::number_of_parameters)
      .def_readonly("maximum_number_of_evaluations", &HillVallEAConfiguration::maximum_number_of_evaluations)
      .def_readonly("maximum_number_of_seconds", &HillVallEAConfiguration::maximum_number_of_seconds)
      .def_readonly("lower", &HillVallEAConfiguration::lower)
      .def_readonly("upper", &HillVallEAConfiguration::upper)
      .def_readonly("random_seed", &HillVallEAConfiguration::random_seed)
      .def_readonly("internal_optimizer_name", &HillVallEAConfiguration::internal_optimizer_name)
      .def_readonly("problem_name", &HillVallEAConfiguration::problem_name)
      .def_readonly("value_to_reach", &HillVallEAConfiguration::value_to_reach)
      .def_readonly("use_value_to_reach", &HillVallEAConfiguration::use_value_to_reach)
      .def_readonly("TargetTolFun", &HillVallEAConfiguration::TargetTolFun);

  m.def("optimize", &optimize, py::arg("config"), "Run HillVallEA and return elitist solutions.");
}
