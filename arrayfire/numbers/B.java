
package arrayfire.numbers;

public record B(int size) implements Num<B> {

    @Override
    public B create(int size) {
        return new B(size);
    }
}
  
