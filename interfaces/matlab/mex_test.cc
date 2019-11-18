#include "mex.hpp"

#include <Eigen/Dense>
#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "mexAdapter.hpp"

using namespace matlab::data;

class MexFunction : public matlab::mex::Function {
 public:
  MexFunction() { MatlabPointer_ = getEngine(); }

  void operator()(matlab::mex::ArgumentList outputs,
                  matlab::mex::ArgumentList inputs) {
    ValidateInputs(inputs);
    // ValidateOutputs

    // Move the input into a CellArray
    CellArray data = std::move(inputs[0]);

    int N = data.getNumberOfElements();

    std::vector<Eigen::MatrixXd> A;
    A.resize(N);

    for (int i = 0; i < N; i++) {
      int n = data[i].getNumberOfElements();
      ArrayDimensions dim = data[i].getDimensions();
      // should make sure that dim is the right length
      // will do error checking in matlab wrapper rather than here
      int nrows = dim.at(0);
      int ncols = dim.at(1);

      // Build an Eigen map over the data
      // .release() returns a unique_ptr, .get() returns a regular pointer
      // If I don't save the unique_ptr will it go out of scope????
      TypedArray<double> r = data[i];
      Eigen::Map<Eigen::MatrixXd> temp(r.release().get(), nrows, ncols);
      // this is somewhat wasteful since the data will be copied
      // once FBstab supports other Eigen types
      A.at(i) = temp;
    }

    for (int i = 0; i < N; i++) {
      std::ostringstream stream;
      stream << A.at(i) << std::endl;
      MatlabMessage(stream);
    }

    StructArray data2 = std::move(inputs[1]);
    auto fields = data2.getFieldNames();
    std::vector<MATLABFieldIdentifier> field_names(fields.begin(),
                                                   fields.end());

    // I can loop over field_names and work on the correct bit of input data by
    // checking for equivalence.
    for (const auto& name : field_names) {
      std::ostringstream stream;
      stream << std::string(name) << std::endl;
      MatlabMessage(stream);
      matlab::data::MATLABFieldIdentifier tmp("A");
      if (name == tmp) {
        MatlabMessage("Found A!\n");
        // data2[0] is a Struct which can be indexed by its field names
        CellArray Adata = std::move(data2[0]["A"]);
        int N2 = Adata.getNumberOfElements();
      }
    }

    // String checking
    CharArray ts = inputs[2];

    TypedArray<char16_t> tt = ts;
    std::u16string str(tt.release().get());
    std::u16string hi_str(u"hi");

    int isHI = str.compare(hi_str);
    std::ostringstream stream;
    stream << isHI << std::endl;
    MatlabMessage(stream);
  }

 private:
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

  void ValidateInputs(matlab::mex::ArgumentList inputs) {
    ArrayFactory factory;
    if (inputs.size() != 3) {
      MatlabError("Expects 3 inputs\n");
    }
    if (inputs[0].getType() != ArrayType::CELL) {
      MatlabError("First input must be a cell\n");
    }
  }

  std::shared_ptr<matlab::engine::MATLABEngine> MatlabPointer_;
};