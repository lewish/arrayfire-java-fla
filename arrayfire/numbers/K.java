
package arrayfire.numbers;

public record K(int size) implements Num<K> {

    @Override
    public K create(int size) {
        return new K(size);
    }
}
  
