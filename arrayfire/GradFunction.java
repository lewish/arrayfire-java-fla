package arrayfire;

import arrayfire.DataType;
import arrayfire.Tensor;
import arrayfire.TensorPair;
import arrayfire.numbers.Num;

import java.util.List;

@FunctionalInterface
interface GradFunction {

    List<Tensor<?, ?, ?, ?, ?>> grads(Tensor<?, ?, ?, ?, ?> resultGrads);

    interface Unary<RT extends DataType<?, ?>, RD0 extends Num<?>, RD1 extends Num<?>, RD2 extends Num<?>, RD3 extends Num<?>, I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>> {
        Tensor<I0T, I0D0, I0D1, I0D2, I0D3> grads(Tensor<RT, RD0, RD1, RD2, RD3> result,
                                                  Tensor<RT, RD0, RD1, RD2, RD3> grads);
    }

    interface Binary<RT extends DataType<?, ?>, RD0 extends Num<?>, RD1 extends Num<?>, RD2 extends Num<?>, RD3 extends Num<?>, I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>> {
        TensorPair<I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> grads(
            Tensor<RT, RD0, RD1, RD2, RD3> result, Tensor<RT, RD0, RD1, RD2, RD3> grads);
    }
}
