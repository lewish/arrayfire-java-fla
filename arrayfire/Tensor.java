package arrayfire;

import arrayfire.datatypes.AfDataType;
import arrayfire.datatypes.B8;
import arrayfire.datatypes.U32;
import arrayfire.numbers.N;
import arrayfire.numbers.U;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import static arrayfire.ArrayFire.af;

public class Tensor<T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> implements TensorLike<T, D0, D1, D2, D3> {

  // Contains a single device pointer.
  public static final AddressLayout LAYOUT = ValueLayout.ADDRESS;

  private Scope scope;
  private final MemorySegment segment;
  private final T type;
  private final Shape<D0, D1, D2, D3> shape;
  private HostTensor<T, D0, D1, D2, D3> data;

  Tensor(Scope scope, MemorySegment segment, T type, Shape<D0, D1, D2, D3> shape) {
    this.scope = scope;
    this.segment = segment;
    this.type = type;
    this.shape = shape;
    this.scope.track(this);
    if (ArrayFire.EAGER.get()) {
      assert Arrays.equals(af.getDims(segment), shape.dims()) : "Tensor shape mismatch";
      this.data = af.data(this);
    } else {
      this.data = null;
    }
    if (ArrayFire.CHECK_NANS.get()) {
      if (type == AfDataType.F32 || type == AfDataType.F64) {
        af.nancheck(this);
      }
    }
  }

  public void setScope(Scope scope) {
    this.scope.untrack(this);
    scope.track(this);
    if (data != null) {
      this.scope.untrack(data);
      scope.track(data);
    }
    this.scope = scope;
  }

  public MemorySegment segment() {
    return segment;
  }

  /**
   * @return the wrapped void* pointer of the C af_array.
   */
  public MemorySegment dereference() {
    //    return MemorySegment.ofAddress(segment.get(LAYOUT,0L));
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

  public Scope scope() {
    return scope;
  }


  public Tensor<T, D1, D0, D2, D3> transpose() {
    return af.transpose(this);
  }

  public <OD0 extends Number> Tensor<T, OD0, D1, D2, D3> castshape(Function<Integer, OD0> d0) {
    return af.castshape(this, d0);
  }

  public <OD0 extends Number, OD1 extends Number> Tensor<T, OD0, OD1, D2, D3> castshape(Function<Integer, OD0> d0,
      Function<Integer, OD1> d1) {
    return af.castshape(this, d0, d1);
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number> Tensor<T, OD0, OD1, OD2, D3> castshape(Function<Integer, OD0> d0,
      Function<Integer, OD1> d1,
      Function<Integer, OD2> d2) {
    return af.castshape(this, d0, d1, d2);
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> castshape(
      Function<Integer, OD0> d0,
      Function<Integer, OD1> d1,
      Function<Integer, OD2> d2,
      Function<Integer, OD3> d3) {
    return af.castshape(this, d0, d1, d2, d3);
  }

  public <OD0 extends Number> Tensor<T, OD0, U, U, U> reshape(OD0 d0) {
    return af.reshape(this, af.shape(d0));
  }

  public <OD0 extends Number, OD1 extends Number> Tensor<T, OD0, OD1, U, U> reshape(OD0 d0, OD1 d1) {
    return af.reshape(this, af.shape(d0, d1));
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number> Tensor<T, OD0, OD1, OD2, U> reshape(OD0 d0,
      OD1 d1,
      OD2 d2) {
    return af.reshape(this, af.shape(d0, d1, d2));
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> reshape(
      OD0 d0,
      OD1 d1,
      OD2 d2,
      OD3 d3) {
    return af.reshape(this, af.shape(d0, d1, d2, d3));
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> reshape(
      Shape<OD0, OD1, OD2, OD3> newShape) {
    return af.reshape(this, newShape);
  }

  public void release() {
    af.release(this);
  }

  public Tensor<T, D0, D1, D2, D3> escape() {
    return af.escape(this);
  }

  public Tensor<T, D0, D1, D2, D3> escape(Scope scope) {
    return af.escape(this, scope);
  }

  public Tensor<T, D0, D1, D2, D3> eval() {
    return af.eval(this);
  }

  public Tensor<T, U, D1, D2, D3> sum() {
    return af.sum(this);
  }

  public Tensor<T, D0, U, D2, D3> sum(arrayfire.dims.D1 dim) {
    return af.sum(this, dim);
  }

  public Tensor<B8, U, D1, D2, D3> sumB8() {
    return af.sumB8(this);
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

  public Tensor<U32, U, D1, D2, D3> imax() {
    return af.imax(this);
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

  public Tensor<T, D0, D1, D2, D3> softmax() {
    return af.softmax(this);
  }

  public Tensor<T, D0, D1, D2, D3> softmax(double temperature) {
    return af.softmax(this, temperature);
  }

  public Tensor<T, D0, D1, D2, D3> sigmoid() {
    return af.sigmoid(this);
  }

  public Tensor<T, D0, D1, D2, D3> sparse(AfStorage storage) {
    return af.sparse(this, storage);
  }

  public Tensor<T, N, U, U, U> index(AfIndex index) {
    return af.index(this, index);
  }

  public Tensor<T, N, N, U, U> index(AfIndex i0, AfIndex i1) {
    return af.index(this, i0, i1);
  }

  public Tensor<T, D0, N, U, U> index(AfIndex i0, AfSpan span) {
    return af.index(this, i0, span);
  }

  public Tensor<T, D0, N, U, U> index(AfSpan span, AfIndex i1) {
    return af.index(this, span, i1);
  }

  public Tensor<T, D0, D1, N, U> index(AfSpan span0, AfSpan span1, AfIndex i2) {
    return af.index(this, span0, span1, i2);
  }


  public Tensor<T, ?, ?, ?, ?> index(AfIndex... indexes) {
    return af.index(this, indexes);
  }


  public <BDT extends Number> List<Tensor<T, D0, BDT, U, U>> batch2(Function<Integer, BDT> type, int batchSize) {
    return af.batch(this, type, batchSize);
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
      TensorLike<T, OD0, OD1, OD2, OD3> newShapeTensor) {
    return af.tileAs(this, newShapeTensor.tensor().shape());
  }

  public <OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
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

  private long[] nativeDims() {
    return af.nativeDims(shape()).toArray(ValueLayout.JAVA_LONG);
  }

  private Shape<?, ?, ?, ?> nativeShape() {
    var dims = nativeDims();
    return af.shape((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
  }

  public <TN extends AfDataType<?>> Tensor<TN, D0, D1, D2, D3> cast(TN t) {
    return af.cast(this, t);
  }

  /**
   * L2 norm.
   */
  public Tensor<T, U, D1, D2, D3> norm() {
    return af.norm(this);
  }

  /**
   * Normalize by dividing by the L2 norm.
   */
  public Tensor<T, D0, D1, D2, D3> normalize() {
    return af.normalize(this);
  }

  public Tensor<T, D0, D1, D2, D3> center() {
    return af.center(this);
  }

  @Override
  public Tensor<T, D0, D1, D2, D3> tensor() {
    return this;
  }

  public HostTensor<T, D0, D1, D2, D3> data() {
    return data(false);
  }

  public HostTensor<T, D0, D1, D2, D3> data(boolean pull) {
    if (pull && data == null) {
      data = af.data(this);
    }
    return data;
  }
}
