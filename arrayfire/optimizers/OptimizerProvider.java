package arrayfire.optimizers;

import arrayfire.Optimizer;
import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;

public interface OptimizerProvider {
    <T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> Optimizer<T, D0, D1, D2, D3> get();
}
