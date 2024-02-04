package arrayfire;

public interface Optimizer<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> {

    public void optimize(Params<T, S> params, Array<T, S> gradients);
}
