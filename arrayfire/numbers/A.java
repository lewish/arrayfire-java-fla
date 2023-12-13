package arrayfire.numbers;

public record A(int size) implements IntNumber<A> {
    @Override
    public A create(int size) {
        return new A(size);
    }
}
