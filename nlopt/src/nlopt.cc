#include <iostream>
#include <Eigen/Dense>
#include <algorithm>

#ifdef USE_MATPLOTLIB
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;
#endif

struct reaction_model {
  reaction_model(const Eigen::Vector2d& parameters) : parameters_(parameters) {}

  double operator()(const double s) const {
    return (parameters_[0] * s) / (parameters_[1] + s);
  }
  double operator()(const double s, const double measured_rate) const {
    return measured_rate - operator()(s);
  }
  const Eigen::Vector2d& parameters_;
};

int main()
{
  Eigen::VectorXd S{{0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.74}};
  Eigen::VectorXd Rate{{0.050, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317}};

  Eigen::Vector2d parameters{0.1, 0.0};

  Eigen::VectorXd residuals(7);

  Eigen::Matrix<double, 7, 2> jacobian;

  for (int iterations = 0; iterations < 100; iterations++)
  {
    // Compute Error
    std::transform(S.begin(), S.end(), Rate.begin(), residuals.begin(), reaction_model(parameters));

    // Compute Jacobian // Derivatives
    Eigen::VectorXd residuals_plus(7);
    for (int p_dim = 0; p_dim < 2; ++p_dim)
    {
      Eigen::Vector2d parameters_plus(parameters);
      const double epsilon = 1e-5;
      parameters_plus[p_dim] += epsilon;
      std::transform(S.begin(), S.end(), Rate.begin(), residuals_plus.begin(), reaction_model(parameters_plus));

      jacobian.col(p_dim) = (residuals_plus - residuals) / epsilon;
    }

    Eigen::Matrix<double, 2, 2> hessian = jacobian.transpose() * jacobian;
    Eigen::Matrix<double, 2, 1> b = jacobian.transpose() * residuals;

    // Add small damping factor for cool animation
    Eigen::Matrix<double, 2, 2> diag = 3 * hessian.diagonal().asDiagonal();

    // Ax = b -> x = A^-1 * x
    Eigen::Vector2d delta = (hessian + diag).ldlt().solve(-b);

    parameters = parameters + delta;
    std::cout << "Iteration # " << iterations << " ";
    std::cout << "New parameters : " << parameters.transpose() << "\n";

    // Plot stuff
    #ifdef USE_MATPLOTLIB
    const size_t curve_elements = 100;
    plt::clf();
    std::vector<double> x_data(curve_elements), y_data(curve_elements);
    double init = 0.0;
    for (double& x : x_data) { x = init+=0.05;};
    std::transform(x_data.begin(), x_data.end(), y_data.begin(), reaction_model(parameters));
    plt::plot(x_data, y_data, "r--");
    std::vector<double> s_map(S.begin(), S.end());
    std::vector<double> rate_map(Rate.begin(), Rate.end());
    plt::plot(s_map, rate_map, ".");
    plt::pause(.05);
    #endif
  }

  std::cout << parameters << "\n";
}