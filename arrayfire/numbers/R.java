
package arrayfire.numbers;

public record R(int size) implements Num<R> {

    @Override
    public R create(int size) {
        return new R(size);
    }
}
  
