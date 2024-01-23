package arrayfire;

import arrayfire.numbers.Num;

public interface Optimizer<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> {

    public void optimize(Params<T, D0, D1, D2, D3> params, Tensor<T, D0, D1, D2, D3> gradients);
}
