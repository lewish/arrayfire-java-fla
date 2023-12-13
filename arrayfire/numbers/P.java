package arrayfire.numbers;

public record P(int size) implements IntNumber<P> {

    @Override
    public P create(int size) {
        return new P(size);
    }
}
