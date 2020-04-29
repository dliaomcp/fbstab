#pragma once

#include <Eigen/Dense>

#include "fbstab/components/abstract_components.h"
#include "fbstab/components/full_variable.h"
#include "tools/copyable_macros.h"

namespace fbstab {

/**
 * This class detects infeasibility in quadratic programs, see
 * mpc_data.h for a description of the QPs.
 * It contains methods for determining if a primal-dual variable
 * is a certificate of either unboundedness (dual infeasibility)
 * or primal infeasibility. It implements
 * Algorithm 3 of https://arxiv.org/pdf/1901.04046.pdf.
 */
class FullFeasibility : public FeasibilityResidual<FullVariable> {
 public:
  FBSTAB_NO_COPY_NO_MOVE_NO_ASSIGN(FullFeasibility)
  /**
   * Allocates workspace memory.
   *
   * @param[in] nz number of decision variables
   * @param[in] nl number of equality constraints
   * @param[in] nv number of inequality constraints
   *
   * Throws a runtime_error if any inputs are non-positive.
   */
  FullFeasibility(int nz, int nl, int nv);

  /**
   * Links to problem data needed to perform calculations.
   * Calculations cannot be performed until a data object is provided.
   * @param[in] data pointer to the problem data
   */
  void LinkData(const Data* data) { data_ = data; }

  /**
   * Checks to see if x is an infeasibility certificate for the QP and stores
   * the result internally.
   * @param[in] x   infeasibility certificate candidate
   * @param[in] tol numerical tolerance
   *
   * Throws a runtime_error if x and *this aren't the same size
   * or if the problem data hasn't been linked.
   */
  FeasibilityStatus CheckFeasibility(const FullVariable& x, double tol);

 private:
  // Workspaces
  Eigen::VectorXd tz_;
  Eigen::VectorXd tl_;
  Eigen::VectorXd tv_;

  const Data* data_ = nullptr;
  void NullDataCheck() const;

  int nz_ = 0;  // number of primal variables
  int nl_ = 0;  // number of equality duals
  int nv_ = 0;  // number of inequality duals
};

}  // namespace fbstab
