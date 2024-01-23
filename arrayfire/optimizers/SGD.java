package arrayfire.optimizers;

import arrayfire.Optimizer;
import arrayfire.Params;
import arrayfire.Tensor;
import arrayfire.af;
import arrayfire.datatypes.DataType;
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

    public <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Optimizer<T, D0, D1, D2, D3> get() {
        return new SGDOptimizer<>();
    }

    public class SGDOptimizer<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> implements Optimizer<T, D0, D1, D2, D3> {

        @Override
        public void optimize(Params<T, D0, D1, D2, D3> params, Tensor<T, D0, D1, D2, D3> gradients) {
            params.set(af.sub(params, af.mul(gradients, learningRate)));
        }
    }
}
