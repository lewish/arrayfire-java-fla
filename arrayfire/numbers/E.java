
package arrayfire.numbers;

public record E(int size) implements Num<E> {

    @Override
    public E create(int size) {
        return new E(size);
    }
}
  
