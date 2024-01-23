
package arrayfire.numbers;

public record W(int size) implements Num<W> {

    @Override
    public W create(int size) {
        return new W(size);
    }
}
  
