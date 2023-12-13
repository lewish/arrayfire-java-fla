package arrayfire.numbers;

public record D(int size) implements IntNumber<D> {
    @Override
    public D create(int size) {
        return new D(size);
    }
}
