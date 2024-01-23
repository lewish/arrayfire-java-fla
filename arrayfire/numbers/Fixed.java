package arrayfire.numbers;

public abstract class Fixed<T> implements Num<T> {
    private final int size;

    public Fixed(int size) {
        this.size = size;
    }

    @Override
    @SuppressWarnings("unchecked")
    public T create(int size) {
        assert size == this.size : "Fixed number must be " + this.size;
        return (T) this;
    }

    @SuppressWarnings("unchecked")
    public T create() {
        return (T) this;
    }

    @Override
    public int size() {
        return size;
    }
}
