for letter in {A..Z}
do
  echo "
package arrayfire.numbers;

public record ${letter}(int size) implements Num<${letter}> {

    @Override
    public ${letter} create(int size) {
        return new ${letter}(size);
    }
}
  " > ${letter}.java
done