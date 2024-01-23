
package arrayfire.numbers;

public record O(int size) implements Num<O> {

    @Override
    public O create(int size) {
        return new O(size);
    }
}
  
