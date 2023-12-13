package arrayfire.numbers;

public record C(int size) implements IntNumber<C> {
    @Override
    public C create(int size) {
        return new C(size);
    }
}
