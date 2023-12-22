package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;
import arrayfire.optimizers.OptimizerProvider;

public class Variable<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> implements TensorLike<T, D0, D1, D2, D3> {
    private Tensor<T, D0, D1, D2, D3> tensor;

    public Variable(Tensor<T, D0, D1, D2, D3> tensor) {
        this.tensor = tensor;
    }

    public void set(Tensor<T, D0, D1, D2, D3> tensor) {
        this.tensor.dispose();
        this.tensor = tensor.move(MemoryScope.scopeOf(this.tensor));
    }

    @Override
    public Tensor<T, D0, D1, D2, D3> tensor() {
        return tensor;
    }
}
