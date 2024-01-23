
package arrayfire.numbers;

public record A(int size) implements Num<A> {

    @Override
    public A create(int size) {
        return new A(size);
    }
}
  
