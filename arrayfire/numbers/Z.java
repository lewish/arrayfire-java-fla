package arrayfire.numbers;

public record Z(int size) implements IntNumber<Z> {
    @Override
    public Z create(int size) {
        return new Z(size);
    }
}
