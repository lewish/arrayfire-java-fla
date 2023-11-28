package arrayfire.utils;

public class Reference<T> {
    private T value;

    public Reference() {
    }

    public Reference(T value) {
        this.value = value;
    }

    public T get() {
        return value;
    }

    public void set(T value) {
        this.value = value;
    }
}
