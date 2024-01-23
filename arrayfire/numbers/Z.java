
package arrayfire.numbers;

public record Z(int size) implements Num<Z> {

    @Override
    public Z create(int size) {
        return new Z(size);
    }
}
  
