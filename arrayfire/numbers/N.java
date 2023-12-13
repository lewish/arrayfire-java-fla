package arrayfire.numbers;

public record N(int size) implements IntNumber<N> {

    @Override
    public N create(int size) {
        return new N(size);
    }
}
