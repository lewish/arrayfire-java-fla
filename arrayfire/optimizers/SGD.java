package arrayfire.optimizers;

import arrayfire.Optimizer;
import arrayfire.Params;
import arrayfire.Tensor;
import arrayfire.af;
import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

public class SGD implements OptimizerProvider {

    public static SGD create() {
        return new SGD();
    }

    private double learningRate = 0.1;

    public SGD learningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public <T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> Optimizer<T, D0, D1, D2, D3> get() {
        return new SGDOptimizer<>();
    }

    public class SGDOptimizer<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> implements Optimizer<T, D0, D1, D2, D3> {

        @Override
        public void optimize(Params<T, D0, D1, D2, D3> params, Tensor<T, D0, D1, D2, D3> gradients) {
            params.set(af.sub(params, af.mul(gradients, learningRate)));
        }
    }
}
