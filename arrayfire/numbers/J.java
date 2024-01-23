
package arrayfire.numbers;

public record J(int size) implements Num<J> {

    @Override
    public J create(int size) {
        return new J(size);
    }
}
  
