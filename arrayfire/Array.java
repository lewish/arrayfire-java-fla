package arrayfire;

import arrayfire.numbers.N;
import arrayfire.numbers.Num;
import arrayfire.numbers.U;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Function;

public class Array<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> implements MemoryContainer {

    // Contains a single device pointer.
    public static final AddressLayout LAYOUT = ValueLayout.ADDRESS;
    private final T type;
    private final S shape;
    private final MemorySegment segment;

    public Array(Prototype<T, S> prototype) {
        this(prototype.type(), prototype.shape());
    }

    Array(T type, S shape) {
        this.type = type;
        this.shape = shape;
        this.segment = Arena.ofAuto().allocate(LAYOUT);
    }

    MemorySegment segment() {
        return segment;
    }

    /**
     * @return the wrapped void* pointer of the C af_array.
     */
    public MemorySegment dereference() {
        var value = segment.get(LAYOUT, 0L);
        if (MemorySegment.NULL.equals(value)) {
            throw new IllegalStateException(
                String.format("Cannot dereference an uninitialized segment (nullptr) %s", shape));
        }
        return value;
    }

    public boolean materialized() {
        return !MemorySegment.NULL.equals(segment.get(LAYOUT, 0L));
    }

    public int capacity() {
        return shape.capacity();
    }

    public S shape() {
        return shape;
    }

    public Prototype<T, S> prototype() {
        return new Prototype<>(type, shape);
    }

    public T type() {
        return type;
    }

    @Override
    public String toString() {
        return "AfTensor{" + "type=" + type + ", shape=" + shape + '}';
    }


    public Array<T, Shape<N, U, U, U>> reshape(int d0) {
        return af.reshape(this, af.shape(d0));
    }

    public Array<T, Shape<N, N, U, U>> reshape(int d0, int d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public Array<T, Shape<N, N, N, U>> reshape(int d0, int d1, int d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public Array<T, Shape<N, N, N, N>> reshape(int d0, int d1, int d2, int d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }

    public <OD0 extends Num<OD0>> Array<T, Shape<OD0, U, U, U>> reshape(OD0 d0) {
        return af.reshape(this, af.shape(d0));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>> Array<T, Shape<OD0, OD1, U, U>> reshape(OD0 d0, OD1 d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>> Array<T, Shape<OD0, OD1, OD2, U>> reshape(
        OD0 d0, OD1 d1, OD2 d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>, OD3 extends Num<OD3>> Array<T, Shape<OD0, OD1, OD2, OD3>> reshape(
        OD0 d0, OD1 d1, OD2 d2, OD3 d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }

    public <NS extends Shape<?, ?, ?, ?>> Array<T, NS> reshape(NS newShape) {
        return af.reshape(this, newShape);
    }

    /**
     * Change the type of the array's D0 dimension to the given type variable provider.
     */
    public <OD0 extends Num<OD0>> Array<T, Shape<OD0, U, U, U>> castshape(Function<Integer, OD0> d0) {
        return af.castshape(this, d0, af::u, af::u, af::u);
    }

    /**
     * Change the type of the array's D0, D1 dimensions to the given type variable providers.
     */
    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>> Array<T, Shape<OD0, OD1, U, U>> castshape(
        Function<Integer, OD0> d0, Function<Integer, OD1> d1) {
        return af.castshape(this, d0, d1, af::u, af::u);
    }

    /**
     * Change the type of the array's D0, D1, D2 dimensions to the given type variable providers.
     */
    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>> Array<T, Shape<OD0, OD1, OD2, U>> castshape(
        Function<Integer, OD0> d0, Function<Integer, OD1> d1, Function<Integer, OD2> d2) {
        return af.castshape(this, d0, d1, d2, af::u);
    }

    /**
     * Change the type of the array's dimensions to the given type variable providers.
     */
    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>, OD3 extends Num<OD3>> Array<T, Shape<OD0, OD1, OD2, OD3>> castshape(
        Function<Integer, OD0> d0, Function<Integer, OD1> d1, Function<Integer, OD2> d2, Function<Integer, OD3> d3) {
        return af.castshape(this, d0, d1, d2, d3);
    }

    public void release() {
        af.release(this);
    }

    Array<T, S> retain() {
        return af.retain(this);
    }


    public Array<T, S> eval() {
        return af.eval(this);
    }

    public Array<T, S> clamp(Array<T, S> lo, Array<T, S> hi) {
        return af.clamp(this, lo, hi);
    }

    public Array<T, S> relu() {
        return af.relu(this);
    }

    public Array<T, S> negate() {
        return af.negate(this);
    }

    public Array<T, S> exp() {
        return af.exp(this);
    }

    public Array<T, S> abs() {
        return af.abs(this);
    }

    public Array<T, S> sqrt() {
        return af.sqrt(this);
    }

    public Array<T, S> sigmoid() {
        return af.sigmoid(this);
    }

    public Array<T, S> sparse(Storage storage) {
        return af.sparse(this, storage);
    }

    public Tileable<T, S> tile() {
        return new Tileable<>(this);
    }

    public <NS extends Shape<?, ?, ?, ?>> Array<T, NS> tileAs(Array<T, NS> newShapeArray) {
        return af.tileAs(this, newShapeArray.shape());
    }

    public <NS extends Shape<?, ?, ?, ?>> Array<T, NS> tileAs(NS newShape) {
        return af.tileAs(this, newShape);
    }

    public Array<T, Shape<N, U, U, U>> flatten() {
        return af.flatten(this);
    }

    public Array<T, S> flip() {
        return af.flip(this);

    }

    public Array<T, S> move(Scope scope) {
        Scope.move(this, scope);
        return this;
    }

    public <TN extends DataType<?>> Array<TN, S> cast(TN t) {
        return af.cast(this, t);
    }

    @Override
    public void dispose() {
        release();
    }
}
