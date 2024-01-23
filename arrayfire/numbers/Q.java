
package arrayfire.numbers;

public record Q(int size) implements Num<Q> {

    @Override
    public Q create(int size) {
        return new Q(size);
    }
}
  
