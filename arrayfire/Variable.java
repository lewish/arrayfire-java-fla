package arrayfire;

/**
 * A variable with an optimizer.
 */
public class Variable<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> extends Tensor<T, S> {

    public Variable(T type, S shape) {
        super(type, shape);
    }

    public void set(Tensor<T, S> tensor) {
        af.set(this, tensor);
    }
}
