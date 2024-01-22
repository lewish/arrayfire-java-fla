package arrayfire.numbers;

public record W(int size) implements IntNumber<W> {
    @Override
    public W create(int size) {
        return new W(size);
    }
}
