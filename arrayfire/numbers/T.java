
package arrayfire.numbers;

public record T(int size) implements Num<T> {

    @Override
    public T create(int size) {
        return new T(size);
    }
}
  
