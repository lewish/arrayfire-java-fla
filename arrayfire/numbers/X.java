
package arrayfire.numbers;

public record X(int size) implements Num<X> {

    @Override
    public X create(int size) {
        return new X(size);
    }
}
  
