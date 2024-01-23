
package arrayfire.numbers;

public record Y(int size) implements Num<Y> {

    @Override
    public Y create(int size) {
        return new Y(size);
    }
}
  
