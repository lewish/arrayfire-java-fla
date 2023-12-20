package arrayfire.autograd;

import arrayfire.Tensor;
import arrayfire.TensorLike;
import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

import java.util.List;

@FunctionalInterface
public interface GradFunction {

    List<Tensor<?, ?, ?, ?, ?>> grads(Tensor<?, ?, ?, ?, ?> resultGrads);

    interface Unary<RT extends DataType<?, ?>, RD0 extends IntNumber<?>, RD1 extends IntNumber<?>, RD2 extends IntNumber<?>, RD3 extends IntNumber<?>, I0T extends DataType<?, ?>, I0D0 extends IntNumber<?>, I0D1 extends IntNumber<?>, I0D2 extends IntNumber<?>, I0D3 extends IntNumber<?>> {
        Tensor<I0T, I0D0, I0D1, I0D2, I0D3> grads(Tensor<RT, RD0, RD1, RD2, RD3> result, Tensor<RT, RD0, RD1, RD2, RD3> grads);
    }

    interface Binary<RT extends DataType<?, ?>, RD0 extends IntNumber<?>, RD1 extends IntNumber<?>, RD2 extends IntNumber<?>, RD3 extends IntNumber<?>, I0T extends DataType<?, ?>, I0D0 extends IntNumber<?>, I0D1 extends IntNumber<?>, I0D2 extends IntNumber<?>, I0D3 extends IntNumber<?>, I1T extends DataType<?, ?>, I1D0 extends IntNumber<?>, I1D1 extends IntNumber<?>, I1D2 extends IntNumber<?>, I1D3 extends IntNumber<?>> {
        TensorPair<I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> grads(
                Tensor<RT, RD0, RD1, RD2, RD3> result, Tensor<RT, RD0, RD1, RD2, RD3> grads);
    }

    record TensorPair<I0T extends DataType<?, ?>, I0D0 extends IntNumber<?>, I0D1 extends IntNumber<?>, I0D2 extends IntNumber<?>, I0D3 extends IntNumber<?>, I1T extends DataType<?, ?>, I1D0 extends IntNumber<?>, I1D1 extends IntNumber<?>, I1D2 extends IntNumber<?>, I1D3 extends IntNumber<?>>(
            Tensor<I0T, I0D0, I0D1, I0D2, I0D3> left, Tensor<I1T, I1D0, I1D1, I1D2, I1D3> right) {

    }
}
