package arrayfire.optimizers;

import arrayfire.Optimizer;
import arrayfire.datatypes.DataType;
import arrayfire.numbers.Num;

public interface OptimizerProvider {
    <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Optimizer<T, D0, D1, D2, D3> get();
}
