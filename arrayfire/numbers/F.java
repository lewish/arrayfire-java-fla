
package arrayfire.numbers;

public record F(int size) implements Num<F> {

    @Override
    public F create(int size) {
        return new F(size);
    }
}
  
