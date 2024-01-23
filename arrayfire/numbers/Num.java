package arrayfire.numbers;

public interface Num<N> {
    int size();
    N create(int size);
}
