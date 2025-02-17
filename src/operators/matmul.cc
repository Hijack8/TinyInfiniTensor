#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0]->getDims();
        int rankA = (int)inputs[0]->getRank();
        auto B = inputs[1]->getDims();
        int rankB = (int)inputs[1]->getRank();

        if (getTransA()) { std::swap(A[rankA - 1], A[rankA - 2]); }
        if (getTransB()) { std::swap(B[rankB - 1], B[rankB - 2]); }
        for (int i = 0; i < rankA - 2; i ++) {
          if (A[i] == 1) A[i] = B[i]; 
        }
        A[rankA - 1] = B[rankB - 1];

        vector<Shape> rst;
        rst.push_back(A);
        return rst;
    }

} // namespace infini
