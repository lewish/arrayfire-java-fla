
package arrayfire.numbers;

public record I(int size) implements Num<I> {

    @Override
    public I create(int size) {
        return new I(size);
    }
}
  
