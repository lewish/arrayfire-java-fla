
package arrayfire.numbers;

public record V(int size) implements Num<V> {

    @Override
    public V create(int size) {
        return new V(size);
    }
}
  
