package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.Num;
import arrayfire.optimizers.OptimizerProvider;

/**
 * A variable with an optimizer.
 */
public class Params<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> implements TensorLike<T, D0, D1, D2, D3> {
    private Tensor<T, D0, D1, D2, D3> tensor;
    private final Optimizer<T, D0, D1, D2, D3> optimizer;

    public Params(Tensor<T, D0, D1, D2, D3> tensor, OptimizerProvider optimizerProvider) {
        this.tensor = tensor;
        this.optimizer = optimizerProvider.get();
    }

    public void set(Tensor<T, D0, D1, D2, D3> tensor) {
        this.tensor.dispose();
        this.tensor = tensor.move(MemoryScope.scopeOf(this.tensor));
    }

    public void optimize(Tensor<T, D0, D1, D2, D3> gradients) {
        if (optimizer == null) {
            throw new IllegalStateException("Attempting to optimize params but no optimizer is provided.");
        }
        optimizer.optimize(this, gradients);
    }

    @Override
    public Tensor<T, D0, D1, D2, D3> tensor() {
        return tensor;
    }
}
