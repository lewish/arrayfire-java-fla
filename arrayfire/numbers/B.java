package arrayfire.numbers;

public record B(int size) implements IntNumber<B> {

    @Override
    public B create(int size) {
        return new B(size);
    }
}
