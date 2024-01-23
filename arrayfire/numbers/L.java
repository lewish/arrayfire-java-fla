
package arrayfire.numbers;

public record L(int size) implements Num<L> {

    @Override
    public L create(int size) {
        return new L(size);
    }
}
  
