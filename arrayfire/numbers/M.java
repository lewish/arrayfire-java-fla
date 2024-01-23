
package arrayfire.numbers;

public record M(int size) implements Num<M> {

    @Override
    public M create(int size) {
        return new M(size);
    }
}
  
