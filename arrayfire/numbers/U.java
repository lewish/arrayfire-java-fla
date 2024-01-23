
package arrayfire.numbers;

public record U(int size) implements Num<U> {

    @Override
    public U create(int size) {
        return new U(size);
    }
}
  
