#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "fbstab/fbstab_dense.h"
#include "mex.hpp"
#include "mexAdapter.hpp"

using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
 public:
  MexFunction() { MatlabPointer_ = getEngine(); }

  /**
   * The first input is a MATLAB string which allows for 3 modes.
   * setup: Call the constructor
   * solve: Call the main solve routine
   * setopts: Update options
   * getopts: Return current options
   * teardown: Call the destructor
   *
   * The other inputs depend on the mode.
   * init(N,nx,nu,nc)
   * solve()
   */
  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {
    ValidateInputs(inputs);
    // ValidateOutputs

    int64_t mode = inputs[0][0];
    std::cout << "Mode = " << mode << std::endl;
    if (val == 0) {
      std::cout << "Setup Mode Active" << std::endl;
    }
  }

 private:
  std::shared_ptr<matlab::engine::MATLABEngine> MatlabPointer_;
  std::unique_ptr<fbstab::FBstabDense> solver_;

  void ValidateInputs(matlab::mex::ArgumentList inputs) {
    ArrayFactory factory;
    if (inputs[0].getType() != ArrayType::INT32) {
      MatlabError("Expects the first input to be an int64\n");
    }
  }

  void MatlabMessage(const std::string& message) {
    ArrayFactory factory;
    MatlabPointer_->feval(u"fprintf", 0,
                          std::vector<Array>({factory.createScalar(message)}));
  }

  void MatlabMessage(const std::ostringstream& stream) {
    ArrayFactory factory;
    MatlabPointer_->feval(
        u"fprintf", 0,
        std::vector<Array>({factory.createScalar(stream.str())}));
  }

  void MatlabError(const std::string& message) {
    ArrayFactory factory;
    MatlabPointer_->feval(u"error", 0,
                          std::vector<Array>({factory.createScalar(message)}));
  }
};