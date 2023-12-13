package arrayfire.numbers;

public record K(int size) implements IntNumber<K> {
    @Override
    public K create(int size) {
        return new K(size);
    }
}
