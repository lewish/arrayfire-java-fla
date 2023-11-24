package arrayfire;

import arrayfire.capi.arrayfire_h;
import arrayfire.datatypes.*;
import arrayfire.numbers.*;
import fade.context.Context;
import fade.context.Contextual;
import fade.contextuals.Contextuals;
import fade.functional.Functions;

import java.lang.foreign.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class ArrayFire {

  public static final Contextual<AfBackend> DEFAULT_BACKEND = Contextual.named("default_backend", AfBackend.DEFAULT);
  public static final Contextual<Boolean> EAGER = Contextual.named("arrayfire_eager", false);
  public static final Contextual<Boolean> CHECK_NANS = Contextual.named("arrayfire_check_nans", false);

  public static final ArrayFire af = new ArrayFire();

  public final arrayfire.dims.D0 d0 = new arrayfire.dims.D0();
  public final arrayfire.dims.D1 d1 = new arrayfire.dims.D1();
  public final arrayfire.dims.D2 d2 = new arrayfire.dims.D2();
  public final arrayfire.dims.D3 d3 = new arrayfire.dims.D3();

  public static void loadNativeLibraries() {
    var libraries = List.of("af", "afcuda", "afopencl", "afcpu");
    for (var library : libraries) {
      try {
        System.loadLibrary(library);
        return;
      } catch (Throwable ignored) {
      }
    }
    throw new RuntimeException("Failed to load ArrayFire native libraries, make sure it is installed.");
  }

  protected ArrayFire() {
    loadNativeLibraries();
    scope(() -> {
      var version = version();
      if (version.major() < 3 || (version.major() == 3 && version.minor() < 8)) {
        throw new IllegalStateException(String.format("Unsupported ArrayFire version, must be >= 3.8.0: %s",
            version));
      }
    });
    if (availableBackends().contains(DEFAULT_BACKEND.get())) {
      setBackend(DEFAULT_BACKEND.get());
    }
  }

  public Context.Entry<Scope> scope() {
    var scope = new Scope(Context.has(Scope.CONTEXTUAL) ? currentScope() : null);
    return new Context.Entry<>(Scope.CONTEXTUAL, scope);
  }

  public void scope(Runnable fn) {
    Context.fork(scope(), fn);
  }

  public <T> T scope(Supplier<T> fn) {
    return Context.fork(scope(), fn);
  }

  @SuppressWarnings("unchecked")
  public <T extends Tensor<?, ?, ?, ?, ?>> T tidy(Supplier<T> fn) {
    return Context.fork(scope(), () -> (T) fn.get().escape());
  }

  public Scope currentScope() {
    return Scope.CONTEXTUAL.get();
  }

  public SegmentAllocator allocator() {
    return Arena.ofConfined();
  }

  /**
   * @return a rank 0 shape (scalar).
   */
  public Shape<U, U, U, U> scalarShape() {
    return shape(u());
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<F32, D0, D1, D2, D3> randgF32(
      Shape<D0, D1, D2, D3> shape) {
    return randgF32(shape, Contextuals.seed().random().nextLong());
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<F32, D0, D1, D2, D3> randgF32(
      Shape<D0, D1, D2, D3> shape,
      long seed) {
    var segment = allocator().allocateArray(AfDataType.F32.layout(), shape.capacity());
    var tensor = af.hostTensor(segment, AfDataType.F32, shape);
    var random = new Random(seed);
    for (int i = 0; i < tensor.capacity(); i++) {
      tensor.type().set(segment, i, (float) random.nextGaussian());
    }
    return tensor;
  }


  public <D0 extends Number> HostTensor<U64, D0, U, U, U> range(D0 d0) {
    var tensor = af.hostTensor(AfDataType.U64, shape(d0));
    for (long i = 0; i < tensor.capacity(); i++) {
      set(tensor, (int) i, i);
    }
    return tensor;
  }

  public <DT extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> copy(
      HostTensor<DT, D0, D1, D2, D3> tensor) {
    var newTensor = af.hostTensor(tensor.type(), tensor.shape());
    newTensor.segment().copyFrom(tensor.segment());
    return newTensor;
  }

  public <JT, DT extends AfDataType<JT>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> shuffle(
      HostTensor<DT, D0, D1, D2, D3> tensor) {
    return shuffle(tensor, Contextuals.seed().random().nextLong());
  }

  public <JT, DT extends AfDataType<JT>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> shuffle(
      HostTensor<DT, D0, D1, D2, D3> tensor,
      long seed) {
    var random = new Random(seed);
    var copy = copy(tensor);
    for (int i = copy.capacity(); i > 1; i--) {
      var randomIndex = random.nextInt(i);
      var tmp = get(copy, i - 1);
      set(copy, i - 1, get(copy, randomIndex));
      set(copy, randomIndex, tmp);
    }
    return copy;
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<F32, D0, D1, D2, D3> randomMask(
      Shape<D0, D1, D2, D3> shape,
      long seed,
      double probability) {
    var segment = allocator().allocateArray(AfDataType.F32.layout(), shape.capacity());
    var tensor = hostTensor(segment, AfDataType.F32, shape);
    var random = new Random(seed);
    for (int i = 0; i < tensor.capacity(); i++) {
      tensor.set(tensor.type(), i, random.nextFloat() < probability ? 1.0f : 0.0f);
    }
    return tensor;
  }


  public HostTensor<F32, N, U, U, U> hostTensor(float[] values) {
    return hostTensor(values, shape(IntNumber.n(values.length)));
  }

  public <T extends AfDataType<?>, D2 extends Number, D3 extends Number, D0 extends Number, D1 extends Number> HostTensor<T, D0, D1, D2, D3> hostTensor(
      MemorySegment segment,
      T type,
      Shape<D0, D1, D2, D3> shape) {
    return hostTensor(currentScope(), segment, type, shape);
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<F32, D0, D1, D2, D3> hostTensor(
      float[] values,
      Shape<D0, D1, D2, D3> shape) {
    var type = AfDataType.F32;
    var segment = allocator().allocateArray(type.layout(), values);
    return hostTensor(currentScope(), segment, type, shape);
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<U32, D0, D1, D2, D3> hostTensor(
      int[] values,
      Shape<D0, D1, D2, D3> shape) {
    var type = AfDataType.U32;
    var segment = allocator().allocateArray(type.layout(), values);
    return hostTensor(currentScope(), segment, type, shape);
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<U64, D0, D1, D2, D3> hostTensor(
      long[] values,
      Shape<D0, D1, D2, D3> shape) {
    var type = AfDataType.U64;
    var segment = allocator().allocateArray(type.layout(), values);
    return hostTensor(currentScope(), segment, type, shape);
  }

  public <DT extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> hostTensor(
      DT type,
      Shape<D0, D1, D2, D3> shape) {
    var segment = allocator().allocateArray(type.layout(), shape.capacity());
    return hostTensor(currentScope(), segment, type, shape);
  }

  public <DT extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> hostTensor(
      Scope scope,
      MemorySegment segment,
      DT type,
      Shape<D0, D1, D2, D3> shape) {

    var hostTensor = new HostTensor<>(segment, type, shape);
    scope.track(hostTensor);
    return hostTensor;
  }

  private <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> deviceTensor(
      MemorySegment segment,
      T type,
      Shape<D0, D1, D2, D3> shape) {
    return deviceTensor(currentScope(), segment, type, shape);
  }

  private <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> deviceTensor(
      Scope scope,
      MemorySegment segment,
      T type,
      Shape<D0, D1, D2, D3> shape) {

    return new Tensor<>(scope, segment, type, shape);
  }

  public Tensor<F32, U, U, U, U> constant(float value) {
    return constant(value, shape(af.u()), AfDataType.F32);
  }

  public Tensor<F64, U, U, U, U> constant(double value) {
    return constant(value, shape(af.u()), AfDataType.F64);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> constant(
      double value,
      TensorLike<T, D0, D1, D2, D3> tensor) {
    return constant(value, tensor.shape(), tensor.tensor().type());
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<F32, D0, D1, D2, D3> constant(
      double value,
      Shape<D0, D1, D2, D3> shape) {
    return constant(value, shape, AfDataType.F32);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> constant(
      double value,
      Shape<D0, D1, D2, D3> shape,
      T type) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_constant(result, value, shape.dims().length, nativeDims(shape), type.code()));
    return deviceTensor(result, type, shape);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> push(
      HostTensor<T, D0, D1, D2, D3> data) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_create_array(result,
        data.segment(),
        data.shape().dims().length,
        nativeDims(data.shape()),
        data.type().code()));
    return deviceTensor(result, data.type(), data.shape());
  }

  public void sync() {
    handleStatus(arrayfire_h.af_sync(deviceId()));
  }

  public AfIndex seq(int begin, int endInclusive, int step) {
    return new AfIndex(new AfSeq(begin, endInclusive, step));
  }

  public AfIndex seq(int begin, int endInclusive) {
    return new AfIndex(new AfSeq(begin, endInclusive, 1));
  }

  public <D0 extends Number> AfIndex seq(Tensor<U64, D0, U, U, U> index) {
    return new AfIndex(index);
  }

  public AfIndex seq(Number num) {
    return af.seq(0, num.intValue() - 1);
  }

  public AfSpan span() {
    return new AfSpan();
  }

  public Shape<N, U, U, U> shape(int d0) {
    return new Shape<>(af.n(d0), u(), u(), u());
  }

  public <D0 extends Number> Shape<D0, U, U, U> shape(D0 d0) {
    return new Shape<>(d0, u(), u(), u());
  }

  public <D0 extends Number> Shape<D0, N, U, U> shape(D0 d0, int d1) {
    return new Shape<>(d0, af.n(d1), u(), u());
  }

  public <D1 extends Number> Shape<N, D1, U, U> shape(int d0, D1 d1) {
    return new Shape<>(af.n(d0), d1, u(), u());
  }

  public Shape<N, N, U, U> shape(int d0, int d1) {
    return new Shape<>(af.n(d0), af.n(d1), u(), u());
  }

  public <D0 extends Number, D1 extends Number> Shape<D0, D1, U, U> shape(D0 d0, D1 d1) {
    return new Shape<>(d0, d1, u(), u());
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number> Shape<D0, D1, D2, U> shape(D0 d0, D1 d1, D2 d2) {
    return new Shape<>(d0, d1, d2, u());
  }

  public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Shape<D0, D1, D2, D3> shape(D0 d0,
                                                                                                                  D1 d1,
                                                                                                                  D2 d2,
                                                                                                                  D3 d3) {
    return new Shape<>(d0, d1, d2, d3);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> reduce(
      Tensor<?, D0, D1, D2, D3> a,
      Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
      T resultType) {
    return reduce(a, method, d0, resultType);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> reduce(
      Tensor<?, D0, D1, D2, D3> a,
      Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
      arrayfire.dims.D0 dim,
      T resultType) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(method.apply(result, a.dereference(), dim.index()));
    return deviceTensor(result, resultType, shape(u(), a.shape().d1(), a.shape().d2(), a.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> reduce(
      Tensor<?, D0, D1, D2, D3> a,
      Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
      arrayfire.dims.D1 dim,
      T resultType) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(method.apply(result, a.dereference(), dim.index()));
    return deviceTensor(result, resultType, shape(a.shape().d0(), u(), a.shape().d2(), a.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, U, D3> reduce(
      Tensor<?, D0, D1, D2, D3> a,
      Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
      arrayfire.dims.D2 dim,
      T resultType) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(method.apply(result, a.dereference(), dim.index()));
    return deviceTensor(result, resultType, shape(a.shape().d0(), a.shape().d1(), u(), a.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, U> reduce(
      Tensor<?, D0, D1, D2, D3> a,
      Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
      arrayfire.dims.D3 dim,
      T resultType) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(method.apply(result, a.dereference(), dim.index()));
    return deviceTensor(result, resultType, shape(a.shape().d0(), a.shape().d1(), a.shape().d2(), u()));
  }


  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> cast(
      Tensor<?, D0, D1, D2, D3> a,
      T type) {
    var result = allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_cast(result, a.dereference(), type.code()));
    return deviceTensor(result, type, a.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> ones(
      TensorLike<T, D0, D1, D2, D3> model) {
    return ones(model.tensor().type(), model.tensor().shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> ones(
      T type,
      Shape<D0, D1, D2, D3> shape) {
    return af.constant(1, shape, type);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> zeros(
      TensorLike<T, D0, D1, D2, D3> model) {
    return zeros(model.tensor().type(), model.tensor().shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> zeros(
      T type,
      Shape<D0, D1, D2, D3> shape) {
    return af.constant(0, shape, type);
  }

  public <JT, DT extends AfDataType<JT>, T extends HostTensor<DT, ?, ?, ?, ?>> JT get(T hostTensor, int index) {
    return hostTensor.get(hostTensor.type(), index);
  }

  public <JT, DT extends AfDataType<JT>, T extends HostTensor<DT, ?, ?, ?, ?>> void set(T hostTensor,
                                                                                        int index,
                                                                                        JT value) {
    hostTensor.set(hostTensor.type(), index, value);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<T, D0, D1, D2, D3> data(
      Tensor<T, D0, D1, D2, D3> a) {
    var result = allocator().allocateArray(a.type().layout(), a.capacity());
    handleStatus(arrayfire_h.af_get_data_ptr(result, a.dereference()));
    return hostTensor(currentScope(), result, a.type(), a.shape());
  }

  public float[] dataf32(Tensor<F32, ?, ?, ?, ?> a) {
    var result = allocator().allocateArray(ValueLayout.JAVA_FLOAT, a.capacity());
    handleStatus(arrayfire_h.af_get_data_ptr(result, a.dereference()));
    return result.toArray(ValueLayout.JAVA_FLOAT);
  }

  public long[] datau64(Tensor<U64, ?, ?, ?, ?> a) {
    var result = allocator().allocateArray(ValueLayout.JAVA_LONG, a.capacity());
    handleStatus(arrayfire_h.af_get_data_ptr(result, a.dereference()));
    return result.toArray(ValueLayout.JAVA_LONG);
  }

  public int[] datau32(Tensor<U32, ?, ?, ?, ?> a) {
    var result = allocator().allocateArray(ValueLayout.JAVA_INT, a.capacity());
    handleStatus(arrayfire_h.af_get_data_ptr(result, a.dereference()));
    return result.toArray(ValueLayout.JAVA_INT);
  }

  public long[] getDims(Tensor<?, ?, ?, ?, ?> a) {
    var dims = allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
    handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0),
        dims.asSlice(8),
        dims.asSlice(16),
        dims.asSlice(24),
        a.dereference()));
    return dims.toArray(ValueLayout.JAVA_LONG);
  }

  public long[] getDims(MemorySegment a) {
    var dims = allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
    handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0),
        dims.asSlice(8),
        dims.asSlice(16),
        dims.asSlice(24),
        a.getAtIndex(ValueLayout.ADDRESS, 0)));
    return dims.toArray(ValueLayout.JAVA_LONG);
  }

  public AfVersion version() {
    var result = allocator().allocateArray(ValueLayout.JAVA_INT, 3);
    handleStatus(arrayfire_h.af_get_version(result, result.asSlice(4), result.asSlice(8)));
    var arr = result.toArray(ValueLayout.JAVA_INT);
    return new AfVersion(arr[0], arr[1], arr[2]);
  }

//  public String lastError() {
//    var allocator = allocator();
//    var message = allocator.allocateArray(ValueLayout.JAVA_BYTE, 1024);
//    var messagePtr = allocator.allocate(ValueLayout.ADDRESS, message);
//    var length = allocator.allocate(ValueLayout.JAVA_LONG);
//    try {
//      api().getLastError.invoke(messagePtr.address(), length.address());
//    } catch (Throwable e) {
//      throw new RuntimeException(e);
//    }
//    var messageSegment = MemorySegment.ofAddress(messagePtr.get(ValueLayout.ADDRESS, 0),
//        length.get(ValueLayout.JAVA_LONG, 0),
//        message.());
//    return new String(messageSegment.toArray(ValueLayout.JAVA_BYTE));
//  }

  public Set<AfBackend> availableBackends() {
    var result = allocator().allocate(ValueLayout.JAVA_INT);
    handleStatus(arrayfire_h.af_get_available_backends(result));
    return AfBackend.fromBitmask(result.get(ValueLayout.JAVA_INT, 0));
  }

  public AfBackend backend() {
    var result = allocator().allocate(ValueLayout.JAVA_INT);
    handleStatus(arrayfire_h.af_get_active_backend(result));
    return AfBackend.fromCode(result.get(ValueLayout.JAVA_INT, 0));
  }

  public void setBackend(AfBackend backend) {
    handleStatus(arrayfire_h.af_set_backend(backend.code()));
  }

  public int deviceId() {
    var result = allocator().allocate(ValueLayout.JAVA_INT);
    handleStatus(arrayfire_h.af_get_device(result));
    return result.get(ValueLayout.JAVA_INT, 0);
  }

  public void setDeviceId(int device) {
    handleStatus(arrayfire_h.af_set_device(device));
  }

  public int deviceCount() {
    var result = allocator().allocate(ValueLayout.JAVA_INT);
    handleStatus(arrayfire_h.af_get_device_count(result));
    return result.get(ValueLayout.JAVA_INT, 0);
  }

  public AfDeviceInfo deviceInfo() {
    var allocator = allocator();
    var name = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
    var platform = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
    var toolkit = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
    var compute = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
    handleStatus(arrayfire_h.af_device_info(name, platform, toolkit, compute));
    return new AfDeviceInfo(name.getUtf8String(0),
        platform.getUtf8String(0),
        toolkit.getUtf8String(0),
        compute.getUtf8String(0));
  }

  public MemorySegment nativeDims(Shape<?, ?, ?, ?> shape) {
    // TODO: These never get cleaned up.
    return allocator().allocateArray(ValueLayout.JAVA_LONG, shape.dims());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D1, D0, D2, D3> transpose(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_transpose(result, tensor.dereference(), false));
    return af.deviceTensor(result,
        tensor.type(),
        af.shape(tensor.shape().d1(), tensor.shape().d0(), tensor.shape().d2(), tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, OD0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, OD0, D1, D2, D3> castshape(
      Tensor<T, ?, D1, D2, D3> tensor,
      Function<Integer, OD0> d0) {
    return reshape(tensor,
        af.shape(d0.apply(tensor.shape().d0().intValue()),
            tensor.shape().d1(),
            tensor.shape().d2(),
            tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, OD0, OD1, D2, D3> castshape(
      Tensor<T, ?, ?, D2, D3> tensor,
      Function<Integer, OD0> d0,
      Function<Integer, OD1> d1) {
    return reshape(tensor, af.shape(d0.apply(tensor.shape().d0().intValue()),
        d1.apply(tensor.shape().d1().intValue()),
        tensor.shape().d2(),
        tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, D3 extends Number> Tensor<T, OD0, OD1, OD2, D3> castshape(
      Tensor<T, ?, ?, ?, D3> tensor,
      Function<Integer, OD0> d0,
      Function<Integer, OD1> d1,
      Function<Integer, OD2> d2) {
    return reshape(tensor, af.shape(d0.apply(tensor.shape().d0().intValue()),
        d1.apply(tensor.shape().d1().intValue()),
        d2.apply(tensor.shape().d2().intValue()),
        tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> castshape(
      Tensor<T, ?, ?, ?, ?> tensor,
      Function<Integer, OD0> d0,
      Function<Integer, OD1> d1,
      Function<Integer, OD2> d2,
      Function<Integer, OD3> d3) {
    return reshape(tensor, af.shape(d0.apply(tensor.shape().d0().intValue()),
        d1.apply(tensor.shape().d1().intValue()),
        d2.apply(tensor.shape().d2().intValue()),
        d3.apply(tensor.shape().d3().intValue())));
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> reshape(
      Tensor<T, ?, ?, ?, ?> tensor,
      Shape<OD0, OD1, OD2, OD3> newShape) {
    assert tensor.shape().capacity() == newShape.capacity() : String.format("New shape %s doesn't have same capacity as original shape %s",
        newShape,
        tensor.shape());
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_moddims(result, tensor.dereference(), newShape.dims().length, af.nativeDims(newShape)));
    return af.deviceTensor(result, tensor.type(), newShape);
  }

  public static void release(Tensor<?, ?, ?, ?, ?> tensor) {
    if (tensor.segment().scope().isAlive()) {
      arrayfire_h.af_release_array(tensor.dereference());
// TODO: Track the allocator and close it.
//      tensor.segment().session().close();
      if (tensor.data() != null) {
        release(tensor.data());
      }
    }
  }


  public static void release(HostTensor<?, ?, ?, ?, ?> tensor) {
    // TODO: Fix release
//    if (tensor.segment().session().isAlive()) {
//      tensor.segment().session().close();
//    }
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> escape(
      Tensor<T, D0, D1, D2, D3> tensor) {
    assert af.currentScope().parent != null : "Can't escape a tensor that doesn't have a parent scope.";
    return escape(tensor, af.currentScope().parent);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> escape(
      Tensor<T, D0, D1, D2, D3> tensor,
      Scope scope) {
    tensor.setScope(scope);
    return tensor;
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> eval(
      Tensor<T, D0, D1, D2, D3> tensor) {
    handleStatus(arrayfire_h.af_eval(tensor.dereference()));
    return tensor;
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> mul(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, ?, ?, ?, ?> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_mul(result, tensor.dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> div(
      TensorLike<T, D0, D1, D2, D3> tensor,
      TensorLike<T, ?, ?, ?, ?> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_div(result, tensor.tensor().dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, tensor.tensor().type(), tensor.tensor().shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> add(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, D0, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_add(result, tensor.dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sub(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, D0, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_sub(result, tensor.dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<B8, D0, D1, D2, D3> gte(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, D0, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_ge(result, tensor.dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, AfDataType.B8, tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> max(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, D0, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_maxof(result, tensor.dereference(), rhs.tensor().dereference(), true));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> min(
      Tensor<T, D0, D1, D2, D3> tensor,
      TensorLike<T, D0, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    var response = arrayfire_h.af_minof(result, tensor.dereference(), tensor.dereference(), true);
    handleStatus(response);
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }


  public <T extends AfDataType<?>, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, D1, D2, D3> join(
      Tensor<T, ?, D1, D2, D3> lhs,
      TensorLike<T, ?, D1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_join(result, 0, lhs.dereference(), rhs.tensor().dereference()));
    return af.deviceTensor(result,
        lhs.type(),
        af.shape(af.n(lhs.shape().d0().intValue() + rhs.tensor().shape().d0().intValue()),
            lhs.shape().d1(),
            lhs.shape().d2(),
            lhs.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, D2, D3> join(
      Tensor<T, D0, ?, D2, D3> lhs,
      TensorLike<T, D0, ?, D2, D3> rhs,
      arrayfire.dims.D1 ignored) {
    assert lhs.shape().d0().intValue() == rhs.tensor().shape().d0().intValue() &&
        lhs.shape().d2().intValue() == rhs.tensor().shape().d2().intValue() &&
        lhs.shape().d3().intValue() == rhs.tensor().shape().d3().intValue() : String.format("Incompatible shapes to join along d1: %s, %s",
        lhs.shape(),
        rhs.tensor().shape());
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_join(result, 1, lhs.dereference(), rhs.tensor().dereference()));
    return af.deviceTensor(result, lhs.type(), af.shape(lhs.shape().d0(),
        af.n(lhs.shape().d1().intValue() + rhs.tensor().shape().d1().intValue()),
        lhs.shape().d2(),
        lhs.shape().d3()));
  }


  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> sum(
      Tensor<T, D0, D1, D2, D3> tensor,
      arrayfire.dims.D1 dim) {
    assert !Set.of(AfDataTypeEnum.B8.code(),
            AfDataTypeEnum.S16.code(),
            AfDataTypeEnum.U8.code(),
            AfDataTypeEnum.U16.code())
        .contains(tensor.type().code()) : "Type will be changed, used type specific method instead.";
    return af.reduce(tensor, arrayfire_h::af_sum, dim, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> sum(
      Tensor<T, D0, D1, D2, D3> tensor) {
    assert !Set.of(AfDataTypeEnum.B8.code(),
            AfDataTypeEnum.S16.code(),
            AfDataTypeEnum.U8.code(),
            AfDataTypeEnum.U16.code())
        .contains(tensor.type().code()) : "Type will be changed, used type specific method instead.";
    return reduce(tensor, arrayfire_h::af_sum, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<B8, U, D1, D2, D3> sumB8(
      Tensor<T, D0, D1, D2, D3> tensor) {
    assert AfDataTypeEnum.B8.code() ==
        tensor.type().code() : "Type will be changed, used type specific method instead.";
    return af.reduce(tensor, arrayfire_h::af_sum, AfDataType.B8);
  }


  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> mean(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return af.reduce(tensor, arrayfire_h::af_mean, d0, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> mean(
      Tensor<T, D0, D1, D2, D3> tensor,
      arrayfire.dims.D0 dim) {
    return af.reduce(tensor, arrayfire_h::af_mean, dim, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> mean(
      Tensor<T, D0, D1, D2, D3> tensor,
      arrayfire.dims.D1 dim) {
    return af.reduce(tensor, arrayfire_h::af_mean, dim, tensor.type());
  }


  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> median(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return af.reduce(tensor, arrayfire_h::af_median, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> max(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return af.reduce(tensor, arrayfire_h::af_max, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> max(
      Tensor<T, D0, D1, D2, D3> tensor,
      arrayfire.dims.D1 dim) {
    return af.reduce(tensor, arrayfire_h::af_max, dim, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> min(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return af.reduce(tensor, arrayfire_h::af_min, tensor.type());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<U32, U, D1, D2, D3> imax(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var allocator = allocator();
    var result = allocator.allocate(ValueLayout.ADDRESS);
    var resultIdx = allocator.allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_imax(result, resultIdx, tensor.dereference(), 0));
    return af.deviceTensor(resultIdx,
        AfDataType.U32,
        shape(IntNumber.U, tensor.shape().d1(), tensor.shape().d2(), tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, K extends Number> TopKResult<T, K, D1, D2, D3> topk(
      Tensor<T, D0, D1, D2, D3> tensor,
      K k) {
    var allocator = allocator();
    var result = allocator.allocate(ValueLayout.ADDRESS);
    var resultIdx = allocator.allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_topk(result, resultIdx, tensor.dereference(), k.intValue(), 0, 0));
    return new TopKResult<>(
        af.deviceTensor(result, tensor.type(), shape(k, tensor.shape().d1(), tensor.shape().d2(), tensor.shape().d3())),
        af.deviceTensor(resultIdx,
            AfDataType.U32,
            shape(k, tensor.shape().d1(), tensor.shape().d2(), tensor.shape().d3())));

  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D0, D2, D3> diag(
      Tensor<T, D0, U, D2, D3> tensor) {
    var allocator = allocator();
    var result = allocator.allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_diag_create(result, tensor.dereference(), 0));
    return af.deviceTensor(result,
        tensor.type(),
        shape(tensor.shape().d0(), tensor.shape().d0(), tensor.shape().d2(), tensor.shape().d3()));
  }

  // https://arrayfire.org/docs/group__blas__func__matmul.htm
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, OD1 extends Number> Tensor<T, D0, OD1, D2, D3> matmul(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, D1, OD1, D2, D3> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    assert tensor.shape().d1().intValue() == rhs.shape().d0().intValue() : String.format("Misaligned shapes for matmul, left: %s right: %s",
        tensor.shape(),
        rhs.shape());
    handleStatus(arrayfire_h.af_matmul(result, tensor.dereference(), rhs.dereference(), 0, 0));
    return af.deviceTensor(result,
        tensor.type(),
        shape(tensor.shape().d0(), rhs.shape().d1(), tensor.shape().d2(), tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> clamp(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, ?, ?> lo,
      Tensor<T, ?, ?, ?, ?> hi) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_clamp(result, tensor.dereference(), lo.dereference(), hi.dereference(), true));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> relu(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return clamp(tensor, af.constant(0f).cast(tensor.type()), af.constant(Float.POSITIVE_INFINITY).cast(tensor.type()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<B8, D0, D1, D2, D3> eq(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, ?, ?> rhs) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_eq(result, tensor.dereference(), rhs.dereference(), true));
    return af.deviceTensor(result, AfDataType.B8, tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> negate(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var minusOne = af.constant(-1f, tensor);
    return mul(tensor, minusOne);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> exp(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_exp(result, tensor.dereference()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> log(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_log(result, tensor.dereference()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> abs(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_abs(result, tensor.dereference()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sqrt(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_sqrt(result, tensor.dereference()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> softmax(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var max = max(tensor);
    var normalized = sub(tensor, max.tileAs(tensor));
    var exp = normalized.exp();
    return div(exp, exp.sum().tileAs(tensor));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> softmax(
      Tensor<T, D0, D1, D2, D3> tensor,
      double temperature) {
    var max = max(tensor);
    var normalized = sub(tensor, max.tileAs(tensor));
    var exp = exp(div(normalized, af.constant((float) temperature).cast(tensor.type())));
    return div(exp, exp.sum().tileAs(tensor));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sigmoid(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var one = af.ones(tensor);
    return div(one, add(one, exp(negate(tensor))));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sparse(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfStorage storage) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_create_sparse_array_from_dense(result, tensor.dereference(), storage.code()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, U, U, U> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfIndex index) {
    return (Tensor<T, N, U, U, U>) index(tensor, new AfIndex[]{index});
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, N, U, U> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfIndex i0,
      AfIndex i1) {
    return (Tensor<T, N, N, U, U>) index(tensor, new AfIndex[]{i0, i1});
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, U, U> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfSpan span,
      AfIndex i1) {
    return (Tensor<T, D0, N, U, U>) index(tensor, new AfIndex[]{af.seq(tensor.shape().d0()), i1});
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, N, U> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfSpan span0,
      AfSpan span1,
      AfIndex i2) {
    return (Tensor<T, D0, D1, N, U>) index(tensor,
        new AfIndex[]{af.seq(tensor.shape().d0()), af.seq(tensor.shape().d1()), i2});
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, U, U> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfIndex i0,
      AfSpan span) {
    return (Tensor<T, D0, N, U, U>) index(tensor, new AfIndex[]{i0, af.seq(tensor.shape().d1())});
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, ?, ?, ?, ?> index(
      Tensor<T, D0, D1, D2, D3> tensor,
      AfIndex... indexes) {
    var allocator = allocator();
    System.out.println(AfIndex.LAYOUT.byteSize());
    var layout = MemoryLayout.sequenceLayout(indexes.length, AfIndex.LAYOUT);
    var nativeIndexes = allocator.allocateArray(AfIndex.LAYOUT, indexes.length);
    for (int i = 0; i < indexes.length; i++) {
      indexes[i].emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(i)),
          AfIndex.LAYOUT.byteSize()));
    }
    var result = allocator.allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_index_gen(result, tensor.dereference(), indexes.length, nativeIndexes));

    // We don't obviously know the new shape, so compute it.
    var dims = allocator.allocateArray(ValueLayout.JAVA_LONG, 4);
    handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0),
        dims.asSlice(8),
        dims.asSlice(16),
        dims.asSlice(24),
        result.get(ValueLayout.ADDRESS, 0)));
    var d0 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 0);
    var d1 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 1);
    var d2 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 2);
    var d3 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 3);
    return af.deviceTensor(result, tensor.type(), af.shape(af.n(d0), af.n(d1), af.n(d2), af.n(d3)));
  }


  // zip two tensors together
  public <LT extends AfDataType<?>, RT extends AfDataType<?>, LD0 extends Number, RD0 extends Number, D1 extends Number> ZipD1<LT, RT, LD0, RD0, D1> zip(
      Tensor<LT, LD0, D1, U, U> left,
      Tensor<RT, RD0, D1, U, U> right) {
    return new ZipD1<>(left, right);
  }

  public <LT extends AfDataType<?>, RT extends AfDataType<?>, LD0 extends Number, RD0 extends Number> List<ZipD1<LT, RT, LD0, RD0, N>> batch(
      ZipD1<LT, RT, LD0, RD0, ?> zip,
      int batchSize) {
    return batch(zip, d1, batchSize);
  }

  public <LT extends AfDataType<?>, RT extends AfDataType<?>, LD0 extends Number, RD0 extends Number> List<ZipD1<LT, RT, LD0, RD0, N>> batch(
      ZipD1<LT, RT, LD0, RD0, ?> zip,
      arrayfire.dims.D1 ignored,
      int batchSize) {
    var left = batch(zip.left(), IntNumber::n, batchSize);
    var right = batch(zip.right(), IntNumber::n, batchSize);
    return IntStream.range(0, left.size()).mapToObj(i -> af.zip(left.get(i), right.get(i))).toList();
  }


  // unbatch
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, D2, D3> unbatch(
      List<Tensor<T, D0, N, D2, D3>> tensors,
      arrayfire.dims.D1 ignored) {
    return tensors.stream().reduce((a, b) -> {
      // TODO: We could do this quicker if we use the variable length join method (up to 10).
      var joined = af.join(a, b, d1);
      a.release();
      b.release();
      return joined;
    }).orElseThrow();
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> List<Tensor<T, D0, N, U, U>> batch(
      Tensor<T, D0, D1, U, U> tensor,
      int batchSize) {
    return batch(tensor, af::n, batchSize);
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, BDT extends Number> List<Tensor<T, D0, BDT, U, U>> batch(
      Tensor<T, D0, D1, D2, D3> tensor,
      Function<Integer, BDT> type,
      int batchSize) {
    var results = new ArrayList<Tensor<T, D0, BDT, U, U>>();
    var d0Seq = af.seq(0, tensor.shape().d0().intValue() - 1);
    for (int i = 0; i < tensor.shape().d1().intValue(); i += batchSize) {
      var computedD1Size = Math.min(batchSize, tensor.shape().d1().intValue() - i);
      var slice = index(tensor, d0Seq, af.seq(i, i + computedD1Size - 1));
      results.add(slice.reshape(shape(tensor.shape().d0(), type.apply(computedD1Size))));
    }
    return results;
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
      Tensor<T, ?, ?, ?, ?> tensor,
      TensorLike<T, OD0, OD1, OD2, OD3> newShapeTensor) {
    return tileAs(tensor, newShapeTensor.tensor().shape());
  }

  public <T extends AfDataType<?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
      Tensor<T, ?, ?, ?, ?> tensor,
      Shape<OD0, OD1, OD2, OD3> newShape) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    assert newShape.capacity() % tensor.shape().capacity() == 0 : "Can't tile perfectly.";
    int d0ratio = newShape.d0().intValue() / tensor.shape().d0().intValue();
    int d1ratio = newShape.d1().intValue() / tensor.shape().d1().intValue();
    int d2ratio = newShape.d2().intValue() / tensor.shape().d2().intValue();
    int d3ratio = newShape.d3().intValue() / tensor.shape().d3().intValue();
    handleStatus(arrayfire_h.af_tile(result, tensor.dereference(), d0ratio, d1ratio, d2ratio, d3ratio));
    return af.deviceTensor(result, tensor.type(), newShape);
  }

  public <T extends AfDataType<?>> Tensor<T, N, U, U, U> flatten(Tensor<T, ?, ?, ?, ?> tensor) {
    return reshape(tensor, shape(tensor.shape().capacity()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, D3, U, U> flatten3(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return reshape(tensor,
        shape(tensor.shape().d0().intValue() * tensor.shape().d1().intValue() * tensor.shape().d2().intValue(),
            tensor.shape().d3()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, N, U, U> flatten2(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return reshape(tensor, shape(tensor.shape().d0().intValue() * tensor.shape().d1().intValue(),
        tensor.shape().d2().intValue() * tensor.shape().d3().intValue()));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> flip(
      Tensor<T, D0, D1, D2, D3> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_flip(result, tensor.dereference(), 0));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }


  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, D2, FD3> filters) {
    return convolve2(tensor, filters, af.shape(1, 1), af.shape(0, 0), af.shape(1, 1));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, D2, FD3> filters,
      Shape<?, ?, ?, ?> stride) {
    return convolve2(tensor, filters, stride, af.shape(0, 0), af.shape(1, 1));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, D2, FD3> filters,
      Shape<?, ?, ?, ?> stride,
      Shape<?, ?, ?, ?> padding) {
    return convolve2(tensor, filters, stride, padding, af.shape(1, 1));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
      Tensor<T, D0, D1, D2, D3> tensor,
      Tensor<T, ?, ?, D2, FD3> filters,
      Shape<?, ?, ?, ?> stride,
      Shape<?, ?, ?, ?> padding,
      Shape<?, ?, ?, ?> dilation) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    // TODO(https://github.com/arrayfire/arrayfire/issues/3402): Convolutions look like they may allocate memory outside of ArrayFire's scope, so sometimes we need to GC first.
    retryWithGc(() -> handleStatus(arrayfire_h.af_convolve2_nn(result,
        tensor.dereference(),
        filters.dereference(),
        2,
        af.nativeDims(stride),
        2,
        af.nativeDims(padding),
        2,
        af.nativeDims(dilation))));
    var computedDims = af.getDims(result);
    return af.deviceTensor(result,
        tensor.type(),
        shape(n((int) computedDims[0]), n((int) computedDims[1]), filters.shape().d3(), tensor.shape().d3()));

  }

  public void nancheck(Tensor<?, ?, ?, ?, ?> tensor) {
    scope(() -> {
      if (tensor.type() instanceof F32 f32) {
        var accessor = f32.accessor(tensor.data(true).segment());
        for (int i = 0; i < tensor.capacity(); i++) {
          if (Float.isNaN(accessor.get(i))) {
            throw new RuntimeException("NaN detected");
          }
        }
      } else if (tensor.type() instanceof F64 f64) {
        var accessor = f64.accessor(tensor.data(true).segment());
        for (int i = 0; i < tensor.capacity(); i++) {
          if (Double.isNaN(accessor.get(i))) {
            throw new RuntimeException("NaN detected");
          }
        }
      } else {
        throw new RuntimeException(String.format("Unsupported type: %s", tensor.type()));
      }
    });
  }

  /**
   * L2 norm.
   */
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> norm(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return sqrt(sum(mul(tensor, tensor)));
  }

  /**
   * Normalize by dividing by the L2 norm.
   */
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> normalize(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return div(tensor, norm(tensor).tileAs(tensor));
  }

  /**
   * Center by subtracting the average.
   */
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> center(
      Tensor<T, D0, D1, D2, D3> tensor) {
    return sub(tensor, mean(tensor).tileAs(tensor));
  }

  // svd
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number> SvdResult<T, D0, D1> svd(Tensor<T, D0, D1, U, U> tensor) {
    var u = af.allocator().allocate(ValueLayout.ADDRESS);
    var s = af.allocator().allocate(ValueLayout.ADDRESS);
    var v = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_svd(u, s, v, tensor.dereference()));

    return new SvdResult<>(af.deviceTensor(u, tensor.type(), shape(tensor.shape().d1(), tensor.shape().d1())),
        af.deviceTensor(s, tensor.type(), af.shape(tensor.shape().d1())),
        af.deviceTensor(v, tensor.type(), shape(tensor.shape().d0(), tensor.shape().d0())));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D0, U, U> cov(Tensor<T, D0, D1, U, U> tensor) {
    var subMean = sub(tensor, mean(tensor, af.d1).tileAs(tensor));
    var matrix = matmul(subMean, subMean.transpose());
    return div(matrix, af.constant(tensor.shape().d1().floatValue() - 1.0f, matrix));
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D0, U, U> zcaMatrix(Tensor<T, D0, D1, U, U> tensor) {
    return af.tidy(() -> {
      var cov = cov(tensor);
      var svd = svd(cov);
      var invSqrtS = diag(div(af.constant(1.0f, svd.s()), sqrt(af.add(svd.s(), af.constant(1e-5f, svd.s())))));
      return matmul(svd.u(), matmul(invSqrtS, svd.u().transpose()));
    });
  }


  public <T extends AfDataType<?>, D extends Number> Tensor<T, D, D, U, U> inverse(TensorLike<T, D, D, U, U> tensor) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_inverse(result, tensor.tensor().dereference(), 0));
    return af.deviceTensor(result, tensor.tensor().type(), tensor.shape());
  }

  // TODO: Add uncropped version.
  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D1, U, U> rotate(Tensor<T, D0, D1, U, U> tensor,
                                                                                                        float angle,
                                                                                                        InterpolationType interpolationType) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_rotate(result, tensor.dereference(), angle, true, interpolationType.code()));
    return af.deviceTensor(result, tensor.type(), tensor.shape());
  }

  public <T extends AfDataType<?>, D0 extends Number, D1 extends Number, ND0 extends Number, ND1 extends Number> Tensor<T, ND0, ND1, U, U> scale(
      Tensor<T, D0, D1, U, U> tensor,
      ND0 nd0,
      ND1 nd1,
      InterpolationType interpolationType) {
    var result = af.allocator().allocate(ValueLayout.ADDRESS);
    handleStatus(arrayfire_h.af_scale(result,
        tensor.dereference(),
        nd0.floatValue() / tensor.d0().floatValue(),
        nd1.floatValue() / tensor.d1().floatValue(),
        nd0.longValue(),
        nd1.longValue(),
        interpolationType.code()));
    return af.deviceTensor(result, tensor.type(), shape(nd0, nd1, tensor.shape().d2(), tensor.shape().d3()));
  }

  public static record DeviceMemInfo(long allocBytes, long allocBuffers, long lockBytes, long lockBuffers) {
  }

  public DeviceMemInfo deviceMemInfo() {
    var allocator = af.allocator();
    var allocBytes = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
    var allocBuffers = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
    var lockBytes = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
    var lockBuffers = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
    handleStatus(arrayfire_h.af_device_mem_info(allocBytes,
        allocBuffers,
        lockBytes,
        lockBuffers));
    return new DeviceMemInfo(allocBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
        allocBuffers.getAtIndex(ValueLayout.JAVA_LONG, 0),
        lockBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
        lockBuffers.getAtIndex(ValueLayout.JAVA_LONG, 0));
  }

  public void printMeminfo() {
    var chars = af.allocator().allocateArray(ValueLayout.JAVA_BYTE, 1);
    handleStatus(arrayfire_h.af_print_mem_info(chars, -1));
  }

  public void deviceGc() {
    handleStatus(arrayfire_h.af_device_gc());
  }

  public N n(int value) {
    return new N(value);
  }

  public A a(int value) {
    return new A(value);
  }

  public B b(int value) {
    return new B(value);
  }

  public C c(int value) {
    return new C(value);
  }

  public P p(int value) {
    return new P(value);
  }


  public K k(int value) {
    return new K(value);
  }


  public U u() {
    return IntNumber.U;
  }

  public U u(int value) {
    return IntNumber.U;
  }

  private void retryWithGc(Runnable fn) {
    try {
      fn.run();
    } catch (ArrayFireException e) {
      if (e.status() == AfStatus.AF_ERR_NO_MEM) {
        deviceGc();
        fn.run();
      } else {
        throw e;
      }
    }
  }

  public Context.Entry<Boolean> eager() {
    return eager(true);
  }

  public Context.Entry<Boolean> eager(boolean eager) {
    return EAGER.create(eager);
  }

  public Context.Entry<Boolean> checkNans() {
    return checkNans(true);
  }

  public Context.Entry<Boolean> checkNans(boolean checkNans) {
    return CHECK_NANS.create(checkNans);
  }

  private long[] nativeDims(Tensor<?, ?, ?, ?, ?> tensor) {
    var dims = af.allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
    handleStatus(arrayfire_h.af_get_dims(dims,
        dims.asSlice(8),
        dims.asSlice(16),
        dims.asSlice(24),
        tensor.dereference()));
    return dims.toArray(ValueLayout.JAVA_LONG);
  }

  private Shape<?, ?, ?, ?> nativeShape(Tensor<?, ?, ?, ?, ?> tensor) {
    var dims = nativeDims(tensor);
    return af.shape((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
  }

  static void handleStatus(Object res) {
    var result = AfStatus.fromCode((int) res);
    if (!AfStatus.AF_SUCCESS.equals(result)) {
      throw new ArrayFireException(result);
      //      String lastError;
      //      try {
      //        lastError = af.lastError();
      //
      //      } catch (Exception e) {
      //        throw new RuntimeException("ArrayFireError: " + result.name());
      //      }
      //      throw new RuntimeException("ArrayFireError: " + result.name() + ": " + lastError);
    }
  }
}
