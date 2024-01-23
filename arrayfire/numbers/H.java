
package arrayfire.numbers;

public record H(int size) implements Num<H> {

    @Override
    public H create(int size) {
        return new H(size);
    }
}
  
