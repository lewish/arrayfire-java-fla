package arrayfire;

import arrayfire.numbers.N;
import arrayfire.numbers.Num;
import arrayfire.numbers.U;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

public class Tensor<T extends DataType<?>, S extends Shape<?, ?, ?, ?>> implements MemoryContainer {

    // Contains a single device pointer.
    public static final AddressLayout LAYOUT = ValueLayout.ADDRESS;
    private final T type;
    private final S shape;
    private final MemorySegment segment;

    public Tensor(Prototype<T, S> prototype) {
        this(prototype.type(), prototype.shape());
    }

    Tensor(T type, S shape) {
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
        return segment.get(LAYOUT, 0L);
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


    public Tensor<T, Shape<N, U, U, U>> reshape(int d0) {
        return af.reshape(this, af.shape(d0));
    }

    public Tensor<T, Shape<N, N, U, U>> reshape(int d0, int d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public Tensor<T, Shape<N, N, N, U>> reshape(int d0, int d1, int d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public Tensor<T, Shape<N, N, N, N>> reshape(int d0, int d1, int d2, int d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }

    public <OD0 extends Num<OD0>> Tensor<T, Shape<OD0, U, U, U>> reshape(OD0 d0) {
        return af.reshape(this, af.shape(d0));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>> Tensor<T, Shape<OD0, OD1, U, U>> reshape(OD0 d0, OD1 d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>> Tensor<T, Shape<OD0, OD1, OD2, U>> reshape(
        OD0 d0, OD1 d1, OD2 d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public <OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>, OD3 extends Num<OD3>> Tensor<T, Shape<OD0, OD1, OD2, OD3>> reshape(
        OD0 d0, OD1 d1, OD2 d2, OD3 d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }

    public <NS extends Shape<?, ? ,? ,?>> Tensor<T, NS> reshape(
        NS newShape) {
        return af.reshape(this, newShape);
    }

    public <NS extends Shape<?, ? ,? ,?>> Tensor<T, NS> castshape(
        NS newShape) {
        if (!Arrays.equals(shape.dims(), newShape.dims())) {
            throw new IllegalArgumentException("Cannot cast shape " + shape + " to " + newShape);
        }
        return af.reshape(this, newShape);
    }

    public void release() {
        af.release(this);
    }

    Tensor<T, S> retain() {
        return af.retain(this);
    }


    public Tensor<T, S> eval() {
        return af.eval(this);
    }

    public Tensor<T, S> clamp(Tensor<T, S> lo, Tensor<T, S> hi) {
        return af.clamp(this, lo, hi);
    }

    public Tensor<T, S> relu() {
        return af.relu(this);
    }

    public Tensor<T, S> negate() {
        return af.negate(this);
    }

    public Tensor<T, S> exp() {
        return af.exp(this);
    }

    public Tensor<T, S> abs() {
        return af.abs(this);
    }

    public Tensor<T, S> sqrt() {
        return af.sqrt(this);
    }

    public Tensor<T, S> sigmoid() {
        return af.sigmoid(this);
    }

    public Tensor<T, S> sparse(Storage storage) {
        return af.sparse(this, storage);
    }

    public Tileable<T, S> tile() {
        return new Tileable<>(this);
    }

    public <NS extends Shape<?, ?, ?, ?>> Tensor<T, NS> tileAs(Tensor<T, NS> newShapeTensor) {
        return af.tileAs(this, newShapeTensor.shape());
    }

    public <NS extends Shape<?, ?, ?, ?>> Tensor<T, NS> tileAs(NS newShape) {
        return af.tileAs(this, newShape);
    }

    public Tensor<T, R1<N>> flatten() {
        return af.flatten(this);
    }

    public Tensor<T, S> flip() {
        return af.flip(this);

    }

    public Tensor<T, S> move(Scope scope) {
        Scope.move(this, scope);
        return this;
    }

    public <TN extends DataType<?>> Tensor<TN, S> cast(TN t) {
        return af.cast(this, t);
    }

    @Override
    public void dispose() {
        release();
    }
}
