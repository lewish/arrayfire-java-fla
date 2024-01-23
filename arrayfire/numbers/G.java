
package arrayfire.numbers;

public record G(int size) implements Num<G> {

    @Override
    public G create(int size) {
        return new G(size);
    }
}
  
