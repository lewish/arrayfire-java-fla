
package arrayfire.numbers;

public record S(int size) implements Num<S> {

    @Override
    public S create(int size) {
        return new S(size);
    }
}
  
