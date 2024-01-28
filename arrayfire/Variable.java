package arrayfire;

import arrayfire.numbers.Num;

/**
 * A variable with an optimizer.
 */
public class Variable<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> extends Tensor<T, D0, D1, D2, D3> {

    public Variable(T type, Shape<D0, D1, D2, D3> shape) {
        super(type, shape);
    }

    public void set(Tensor<T, D0, D1, D2, D3> tensor) {
        af.replace(tensor, this);
    }
}
