package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

public interface Optimizer<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> {

    public void optimize(Params<T, D0, D1, D2, D3> params, Tensor<T, D0, D1, D2, D3> gradients);
}
