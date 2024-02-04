package arrayfire.optimizers;

import arrayfire.*;
import arrayfire.numbers.Num;

public class SGD implements OptimizerProvider {

    public static SGD create() {
        return new SGD();
    }

    private double learningRate = 0.1;

    public SGD learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Optimizer<T, S> get() {
        return new SGDOptimizer<>();
    }

    public class SGDOptimizer<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> implements Optimizer<T, S> {

        @Override
        public void optimize(Params<T, S> params, Tensor<T, S> gradients) {
            params.set(af.sub(params, af.mul(gradients, learningRate)));
        }
    }
}
