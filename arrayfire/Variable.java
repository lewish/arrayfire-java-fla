package arrayfire;

/**
 * A variable with an optimizer.
 */
public class Variable<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> extends Array<T, S> {

    public Variable(T type, S shape) {
        super(type, shape);
    }

    public void set(Array<T, S> array) {
        af.set(this, array);
    }
}
