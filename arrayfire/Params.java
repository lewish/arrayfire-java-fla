package arrayfire;

import arrayfire.numbers.Num;
import arrayfire.optimizers.OptimizerProvider;

/**
 * A variable with an optimizer.
 */
public class Params<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> extends Tensor<T, D0, D1, D2, D3> {

    private final Optimizer<T, D0, D1, D2, D3> optimizer;

    public Params(T type, Shape<D0, D1, D2, D3> shape, OptimizerProvider optimizerProvider) {
        super(type, shape);
        this.optimizer = optimizerProvider.get();
    }

    public void set(Tensor<T, D0, D1, D2, D3> tensor) {
        af.retainInto(tensor, this);
    }

    public void optimize(Tensor<T, D0, D1, D2, D3> gradients) {
        if (optimizer == null) {
            throw new IllegalStateException("Attempting to optimize params but no optimizer is provided.");
        }
        optimizer.optimize(this, gradients);
    }
}
