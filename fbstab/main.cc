#include <iostream>

#include <Eigen/Dense>

int main(void) {

  Eigen::MatrixXd A = Eigen::MatrixXd::Ones(5, 5);

  std::cout << A << "\n";
}
