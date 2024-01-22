package arrayfire.numbers;

public record H(int size) implements IntNumber<H> {
    @Override
    public H create(int size) {
        return new H(size);
    }
}
