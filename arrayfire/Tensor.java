package arrayfire;

import arrayfire.datatypes.DataType;
import arrayfire.numbers.IntNumber;
import arrayfire.numbers.N;
import arrayfire.numbers.U;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.function.Function;

public class Tensor<T extends DataType<?, ?>, D0 extends IntNumber<?>, D1 extends IntNumber<?>, D2 extends IntNumber<?>, D3 extends IntNumber<?>> implements TensorLike<T, D0, D1, D2, D3>, MemoryContainer {

    // Contains a single device pointer.
    public static final AddressLayout LAYOUT = ValueLayout.ADDRESS;

    private final Arena arena;
    private final MemorySegment segment;
    private final T type;
    private final Shape<D0, D1, D2, D3> shape;

    Tensor(T type, Shape<D0, D1, D2, D3> shape) {
        this.arena = Arena.ofShared();
        this.segment = arena.allocate(LAYOUT);
        this.type = type;
        this.shape = shape;
        af.memoryScope().track(this);
    }

    public MemorySegment segment() {
        return segment;
    }

    /**
     * @return the wrapped void* pointer of the C af_array.
     */
    public MemorySegment dereference() {
        return segment.get(LAYOUT, 0L);
    }

    public D0 d0() {
        return shape.d0();
    }

    public D1 d1() {
        return shape.d1();
    }

    public D2 d2() {
        return shape.d2();
    }

    public D3 d3() {
        return shape.d3();
    }

    public int capacity() {
        return shape.capacity();
    }

    public Shape<D0, D1, D2, D3> shape() {
        return shape;
    }

    public T type() {
        return type;
    }

    @Override
    public String toString() {
        return "AfTensor{" + "type=" + type + ", shape=" + shape + '}';
    }

    public Tensor<T, D1, D0, D2, D3> transpose() {
        return af.transpose(this);
    }

    public <OD0 extends IntNumber<?>> Tensor<T, OD0, D1, D2, D3> castshape(Function<Integer, OD0> d0) {
        return af.castshape(this, d0);
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>> Tensor<T, OD0, OD1, D2, D3> castshape(Function<Integer, OD0> d0,
                                                                                          Function<Integer, OD1> d1) {
        return af.castshape(this, d0, d1);
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, D3> castshape(
            Function<Integer, OD0> d0, Function<Integer, OD1> d1, Function<Integer, OD2> d2) {
        return af.castshape(this, d0, d1, d2);
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>, OD3 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, OD3> castshape(
            Function<Integer, OD0> d0, Function<Integer, OD1> d1, Function<Integer, OD2> d2,
            Function<Integer, OD3> d3) {
        return af.castshape(this, d0, d1, d2, d3);
    }

    public Tensor<T, N, U, U, U> reshape(int d0) {
        return af.reshape(this, af.shape(d0));
    }

    public <OD0 extends IntNumber<?>> Tensor<T, OD0, U, U, U> reshape(OD0 d0) {
        return af.reshape(this, af.shape(d0));
    }

    public Tensor<T, N, N, U, U> reshape(int d0, int d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>> Tensor<T, OD0, OD1, U, U> reshape(OD0 d0, OD1 d1) {
        return af.reshape(this, af.shape(d0, d1));
    }

    public Tensor<T, N, N, N, U> reshape(int d0, int d1, int d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public Tensor<T, N, N, N, N> reshape(int d0, int d1, int d2, int d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }


    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, U> reshape(OD0 d0,
                                                                                                            OD1 d1,
                                                                                                            OD2 d2) {
        return af.reshape(this, af.shape(d0, d1, d2));
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>, OD3 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, OD3> reshape(
            OD0 d0, OD1 d1, OD2 d2, OD3 d3) {
        return af.reshape(this, af.shape(d0, d1, d2, d3));
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>, OD3 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, OD3> reshape(
            Shape<OD0, OD1, OD2, OD3> newShape) {
        return af.reshape(this, newShape);
    }

    public void release() {
        ArrayFire.release(this);
    }


    public Tensor<T, D0, D1, D2, D3> eval() {
        return af.eval(this);
    }

    public Tensor<T, U, D1, D2, D3> mean() {
        return af.mean(this);
    }

    public Tensor<T, U, D1, D2, D3> mean(arrayfire.dims.D0 dim) {
        return af.mean(this, dim);
    }

    public Tensor<T, D0, U, D2, D3> mean(arrayfire.dims.D1 dim) {
        return af.mean(this, dim);
    }

    public Tensor<T, U, D1, D2, D3> median() {
        return af.median(this);
    }

    public Tensor<T, U, D1, D2, D3> max() {
        return af.max(this);
    }

    public Tensor<T, D0, U, D2, D3> max(arrayfire.dims.D1 dim) {
        return af.max(this, dim);
    }

    public Tensor<T, U, D1, D2, D3> min() {
        return af.min(this);
    }

    public Tensor<T, D0, D1, D2, D3> clamp(Tensor<T, ?, ?, ?, ?> lo, Tensor<T, ?, ?, ?, ?> hi) {
        return af.clamp(this, lo, hi);
    }

    public Tensor<T, D0, D1, D2, D3> relu() {
        return af.relu(this);
    }

    public Tensor<T, D0, D1, D2, D3> negate() {
        return af.negate(this);
    }

    public Tensor<T, D0, D1, D2, D3> exp() {
        return af.exp(this);
    }

    public Tensor<T, D0, D1, D2, D3> abs() {
        return af.abs(this);
    }

    public Tensor<T, D0, D1, D2, D3> sqrt() {
        return af.sqrt(this);
    }

    public Tensor<T, D0, D1, D2, D3> sigmoid() {
        return af.sigmoid(this);
    }

    public Tensor<T, D0, D1, D2, D3> sparse(Storage storage) {
        return af.sparse(this, storage);
    }

    public Tileable<T, D0, D1, D2, D3> tile() {
        return new Tileable<>(this);
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>, OD3 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
            TensorLike<T, OD0, OD1, OD2, OD3> newShapeTensor) {
        return af.tileAs(this, newShapeTensor.tensor().shape());
    }

    public <OD0 extends IntNumber<?>, OD1 extends IntNumber<?>, OD2 extends IntNumber<?>, OD3 extends IntNumber<?>> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
            Shape<OD0, OD1, OD2, OD3> newShape) {
        return af.tileAs(this, newShape);
    }

    public Tensor<T, N, U, U, U> flatten() {
        return af.flatten(this);
    }

    public Tensor<T, N, N, U, U> flatten2() {
        return af.flatten2(this);
    }

    public Tensor<T, N, D3, U, U> flatten3() {
        return af.flatten3(this);
    }

    public Tensor<T, D0, D1, D2, D3> flip() {
        return af.flip(this);

    }

    public <TN extends DataType<?, ?>> Tensor<TN, D0, D1, D2, D3> cast(TN t) {
        return af.cast(this, t);
    }

    /**
     * Normalize by dividing by the L2 norm.
     */

    public Tensor<T, D0, D1, D2, D3> center() {
        return af.center(this);
    }

    @Override
    public Tensor<T, D0, D1, D2, D3> tensor() {
        return this;
    }

    @Override
    public void dispose() {
        ArrayFire.release(this);
        arena.close();
    }
}
