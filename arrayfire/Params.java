package arrayfire;

import arrayfire.optimizers.OptimizerProvider;

/**
 * A variable with an optimizer.
 */
public class Params<T extends DataType<?>, S extends Shape<?, ? ,? ,?>> extends Variable<T, S> {

    private final Optimizer<T, S> optimizer;

    public Params(T type, S shape, OptimizerProvider optimizerProvider) {
        super(type, shape);
        this.optimizer = optimizerProvider.get();
    }

    public void optimize(Array<T, S> gradients) {
        if (optimizer == null) {
            throw new IllegalStateException("Attempting to optimize params but no optimizer is provided.");
        }
        optimizer.optimize(this, gradients);
    }
}
