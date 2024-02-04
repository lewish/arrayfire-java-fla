package arrayfire.optimizers;

import arrayfire.DataType;
import arrayfire.Optimizer;
import arrayfire.Shape;

public interface OptimizerProvider {
    <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Optimizer<T, S> get();
}
