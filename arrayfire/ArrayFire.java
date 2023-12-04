package arrayfire;

import arrayfire.capi.arrayfire_h;
import arrayfire.containers.NativeArray;
import arrayfire.datatypes.*;
import arrayfire.dims.Dim;
import arrayfire.numbers.*;
import arrayfire.utils.Functions;
import arrayfire.utils.Reference;

import java.lang.foreign.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.IntStream;

public class ArrayFire {
    ThreadLocal<MemoryScope> threadScope = ThreadLocal.withInitial(() -> null);

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
        tidy(() -> {
            var version = version();
            if (version.major() < 3 || (version.major() == 3 && version.minor() < 8)) {
                throw new IllegalStateException(
                        String.format("Unsupported ArrayFire version, must be >= 3.8.0: %s", version));
            }
        });
    }

    /**
     * Executes the given function in a new scope, and disposes of all memory allocated in that scope afterward.
     */
    public void tidy(Runnable fn) {
        var previousScope = threadScope.get();
        try {
            threadScope.set(new MemoryScope());
            fn.run();
            threadScope.get().dispose();
        } finally {
            threadScope.set(previousScope);
        }
    }

    /**
     * Executes the given function in a new scope, and disposes of all memory allocated in that scope except the value returned by the function if it is manually managed memory container.
     */
    public <T> T tidy(Supplier<T> fn) {
        var parentScope = currentScope();
        var resultReference = new Reference<T>();
        tidy(() -> {
            var result = (T) fn.get();
            if (result instanceof MemoryContainer mc) {
                parentScope.track(mc);
                currentScope().untrack(mc);
            }
            resultReference.set(result);
        });
        return resultReference.get();
    }

    public MemoryScope currentScope() {
        return threadScope.get();
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


//  public <JT, DT extends AfDataType<JT>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> shuffle(
//      HostTensor<DT, D0, D1, D2, D3> tensor) {
//    return shuffle(tensor, Contextuals.seed().random().nextLong());
//  }
//
//  public <JT, DT extends AfDataType<JT>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> HostTensor<DT, D0, D1, D2, D3> shuffle(
//      HostTensor<DT, D0, D1, D2, D3> tensor,
//      long seed) {
//    var random = new Random(seed);
//    var copy = copy(tensor);
//    for (int i = copy.capacity(); i > 1; i--) {
//      var randomIndex = random.nextInt(i);
//      var tmp = get(copy, i - 1);
//      set(copy, i - 1, get(copy, randomIndex));
//      set(copy, randomIndex, tmp);
//    }
//    return copy;
//  }


    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor) {
        return sort(tensor, d0, true);
    }

    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sort(tensor, dim, true);
    }

    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sort(ptr, tensor.dereference(), dim.index(), ascending));
    }

    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor) {
        return sortIndex(tensor, d0, true);
    }

    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sortIndex(tensor, dim, true);
    }

    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        var values = new Tensor<>(tensor.type(), tensor.shape());
        var indices = new Tensor<>(DataType.U32, tensor.shape());
        handleStatus(arrayfire_h.af_sort_index(values.segment(), indices.segment(), tensor.dereference(), dim.index(),
                ascending));
        return new SortIndexResult<>(values, indices);
    }

    @SuppressWarnings("unchecked")
    public <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<DT, D0, D1, D2, D3> shuffle(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return tidy(() -> {
            var tmp = randu(DataType.U32, af.shape(af.n((int) tensor.shape().dims()[dim.index()])));
            var indices = sortIndex(tmp).indices();
            var args = new Index[4];
            for (int i = 0; i < args.length; i++) {
                args[i] = i == dim.index() ? seq(indices) : af.seq(tensor.shape().dims()[i]);
            }
            return (Tensor<DT, D0, D1, D2, D3>) index(tensor, args);
        });
    }

    public <DT extends DataType<?, ?>, AT extends NativeArray<DT, ?, ?>> Tensor<DT, N, U, U, U> create(AT array) {
        var shape = af.shape(af.n(array.length()));
        var result = new Tensor<>(array.type(), shape);
        handleStatus(arrayfire_h.af_create_array(result.segment(), array.segment(), 1, nativeDims(shape),
                array.type().code()));
        return result;
    }


    @SafeVarargs
    public final <JT, AT extends NativeArray<DT, JT, ?>, DT extends DataType<AT, ?>> Tensor<DT, N, U, U, U> create(
            DT type, JT... values) {
        var array = type.create(values.length);
        try {
            for (int i = 0; i < values.length; i++) {
                array.set(i, values[i]);
            }
            return create(array);
        } finally {
            array.arena().close();
        }
    }

    @SuppressWarnings("unchecked")
    public final <JT, JTA, AT extends NativeArray<DT, JT, JTA>, DT extends DataType<AT, ?>> Tensor<DT, N, U, U, U> create(
            DT type, JTA values) {
        var length = Array.getLength(values);
        var array = type.create(length);
        try {
            for (int i = 0; i < length; i++) {
                array.set(i, (JT) Array.get(values, i));
            }
            return create(array);
        } finally {
            array.arena().close();
        }
    }

    public Tensor<F32, N, U, U, U> create(float... values) {
        return create(DataType.F32, values);
    }

    public Tensor<F64, N, U, U, U> create(double... values) {
        return create(DataType.F64, values);
    }

    public Tensor<S32, N, U, U, U> create(int... values) {
        return create(DataType.S32, values);
    }

    public Tensor<F32, U, U, U, U> constant(float value) {
        return constant(DataType.F32, value);
    }

    public final <DT extends DataType<?, ?>> Tensor<DT, U, U, U, U> constant(DT type, double value) {
        return constant(type, shape(u()), value);
    }

    public final <DT extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<DT, D0, D1, D2, D3> constant(
            DT type, Shape<D0, D1, D2, D3> shape, double value) {
        var result = new Tensor<>(type, shape);
        handleStatus(
                arrayfire_h.af_constant(result.segment(), value, shape.dims().length, nativeDims(shape), type.code()));
        return result;
    }

    public void sync() {
        handleStatus(arrayfire_h.af_sync(deviceId()));
    }

    public Index seq(int begin, int endInclusive, int step) {
        return new Index(new Seq(begin, endInclusive, step));
    }

    public Index seq(int begin, int endInclusive) {
        return new Index(new Seq(begin, endInclusive, 1));
    }

    public <DT extends DataType<?, ?>, D0 extends Number> Index seq(Tensor<DT, D0, U, U, U> index) {
        return new Index(index);
    }

    public Index seq(Number num) {
        return af.seq(0, num.intValue() - 1);
    }

    public Span span() {
        return new Span();
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

    public Shape<N, N, N, U> shape(int d0, int d1, int d2) {
        return new Shape<>(af.n(d0), af.n(d1), af.n(d2), u());
    }

    public <D0 extends Number, D1 extends Number, D2 extends Number> Shape<D0, D1, D2, U> shape(D0 d0, D1 d1, D2 d2) {
        return new Shape<>(d0, d1, d2, u());
    }

    public <D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Shape<D0, D1, D2, D3> shape(
            D0 d0, D1 d1, D2 d2, D3 d3) {
        return new Shape<>(d0, d1, d2, d3);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> reduce(
            Tensor<?, D0, D1, D2, D3> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
            T resultType) {
        return reduce(a, method, d0, resultType);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> reduce(
            Tensor<?, D0, D1, D2, D3> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
            arrayfire.dims.D0 dim, T resultType) {
        var result = new Tensor<>(resultType, shape(u(), a.shape().d1(), a.shape().d2(), a.shape().d3()));
        handleStatus(method.apply(result.segment(), a.dereference(), dim.index()));
        return result;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> reduce(
            Tensor<?, D0, D1, D2, D3> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
            arrayfire.dims.D1 dim, T resultType) {
        var result = new Tensor<>(resultType, shape(a.shape().d0(), u(), a.shape().d2(), a.shape().d3()));
        handleStatus(method.apply(result.segment(), a.dereference(), dim.index()));
        return result;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, U, D3> reduce(
            Tensor<?, D0, D1, D2, D3> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
            arrayfire.dims.D2 dim, T resultType) {
        var result = new Tensor<>(resultType, shape(a.shape().d0(), a.shape().d1(), u(), a.shape().d3()));
        handleStatus(method.apply(result.segment(), a.dereference(), dim.index()));
        return result;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, U> reduce(
            Tensor<?, D0, D1, D2, D3> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
            arrayfire.dims.D3 dim, T resultType) {
        var result = new Tensor<>(resultType, shape(a.shape().d0(), a.shape().d1(), a.shape().d2(), u()));
        handleStatus(method.apply(result.segment(), a.dereference(), dim.index()));
        return result;
    }


    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> cast(
            Tensor<?, D0, D1, D2, D3> a, T arrayType) {
        var result = new Tensor<>(arrayType, a.shape());
        handleStatus(arrayfire_h.af_cast(result.segment(), a.dereference(), arrayType.code()));
        return result;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> ones(
            TensorLike<T, D0, D1, D2, D3> model) {
        return ones(model.tensor().type(), model.tensor().shape());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> ones(
            T type, Shape<D0, D1, D2, D3> shape) {
        return af.constant(type, 1).tileAs(shape);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> zeros(
            TensorLike<T, D0, D1, D2, D3> model) {
        return zeros(model.tensor().type(), model.tensor().shape());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> zeros(
            T type, Shape<D0, D1, D2, D3> shape) {
        return af.constant(type, 0).tileAs(shape);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> randu(
            T type, Shape<D0, D1, D2, D3> shape) {
        return createFromOperation(type, shape, ptr -> arrayfire_h.af_randu(ptr, shape.dims().length, nativeDims(shape),
                type.code()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> randn(
            T type, Shape<D0, D1, D2, D3> shape) {
        return createFromOperation(type, shape, ptr -> arrayfire_h.af_randn(ptr, shape.dims().length, nativeDims(shape),
                type.code()));
    }

    public Tensor<U32, N, U, U, U> range(int n) {
        var shape = af.shape(n);
        return createFromOperation(DataType.U32, shape,
                ptr -> arrayfire_h.af_range(ptr, shape.dims().length, nativeDims(shape), 0,
                        DataType.U32.code()));
    }

    public <T extends DataType<?, ?>> Tensor<T, N, U, U, U> range(
            T type, int n) {
        var shape = af.shape(af.n(n));
        return createFromOperation(type, shape,
                ptr -> arrayfire_h.af_range(ptr, shape.dims().length, nativeDims(shape), 0,
                        type.code()));
    }

    public void setSeed(long seed) {
        handleStatus(arrayfire_h.af_set_seed(seed));
    }

    public void setRandomEngineType(int type) {
        handleStatus(arrayfire_h.af_set_default_random_engine_type(type));
    }

    public <AT extends NativeArray<?, ?, ?>, T extends DataType<AT, ?>> AT data(Tensor<T, ?, ?, ?, ?> a) {
        var result = a.type().create(a.capacity());
        handleStatus(arrayfire_h.af_get_data_ptr(result.segment(), a.dereference()));
        return result;
    }

    public long[] getDims(Tensor<?, ?, ?, ?, ?> a) {
        var dims = allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
        handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                a.dereference()));
        return dims.toArray(ValueLayout.JAVA_LONG);
    }

    public long[] getDims(MemorySegment a) {
        var dims = allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
        handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                a.getAtIndex(ValueLayout.ADDRESS, 0)));
        return dims.toArray(ValueLayout.JAVA_LONG);
    }

    public Version version() {
        var result = allocator().allocateArray(ValueLayout.JAVA_INT, 3);
        handleStatus(arrayfire_h.af_get_version(result, result.asSlice(4), result.asSlice(8)));
        var arr = result.toArray(ValueLayout.JAVA_INT);
        return new Version(arr[0], arr[1], arr[2]);
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

    public Set<Backend> availableBackends() {
        var result = allocator().allocate(ValueLayout.JAVA_INT);
        handleStatus(arrayfire_h.af_get_available_backends(result));
        return Backend.fromBitmask(result.get(ValueLayout.JAVA_INT, 0));
    }

    public Backend backend() {
        var result = allocator().allocate(ValueLayout.JAVA_INT);
        handleStatus(arrayfire_h.af_get_active_backend(result));
        return Backend.fromCode(result.get(ValueLayout.JAVA_INT, 0));
    }

    public void setBackend(Backend backend) {
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

    public DeviceInfo deviceInfo() {
        var allocator = allocator();
        var name = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
        var platform = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
        var toolkit = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
        var compute = allocator.allocateArray(ValueLayout.JAVA_CHAR, 64);
        handleStatus(arrayfire_h.af_device_info(name, platform, toolkit, compute));
        return new DeviceInfo(name.getUtf8String(0), platform.getUtf8String(0), toolkit.getUtf8String(0),
                compute.getUtf8String(0));
    }

    public MemorySegment nativeDims(Shape<?, ?, ?, ?> shape) {
        // TODO: These never get cleaned up.
        return allocator().allocateArray(ValueLayout.JAVA_LONG, shape.dims());
    }

    private <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> createFromOperation(
            T type, Shape<D0, D1, D2, D3> shape, Function<MemorySegment, Integer> method) {
        var result = new Tensor<>(type, shape);
        handleStatus(method.apply(result.segment()));
        return result;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D1, D0, D2, D3> transpose(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return createFromOperation(tensor.type(),
                af.shape(tensor.shape().d1(), tensor.shape().d0(), tensor.shape().d2(), tensor.shape().d3()),
                ptr -> arrayfire_h.af_transpose(ptr, tensor.dereference(), false));

    }

    public <T extends DataType<?, ?>, OD0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, OD0, D1, D2, D3> castshape(
            Tensor<T, ?, D1, D2, D3> tensor, Function<Integer, OD0> d0) {
        return reshape(tensor,
                af.shape(d0.apply(tensor.shape().d0().intValue()), tensor.shape().d1(), tensor.shape().d2(),
                        tensor.shape().d3()));
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, OD0, OD1, D2, D3> castshape(
            Tensor<T, ?, ?, D2, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1) {
        return reshape(tensor,
                af.shape(d0.apply(tensor.shape().d0().intValue()), d1.apply(tensor.shape().d1().intValue()),
                        tensor.shape().d2(), tensor.shape().d3()));
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, D3 extends Number> Tensor<T, OD0, OD1, OD2, D3> castshape(
            Tensor<T, ?, ?, ?, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
            Function<Integer, OD2> d2) {
        return reshape(tensor,
                af.shape(d0.apply(tensor.shape().d0().intValue()), d1.apply(tensor.shape().d1().intValue()),
                        d2.apply(tensor.shape().d2().intValue()), tensor.shape().d3()));
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> castshape(
            Tensor<T, ?, ?, ?, ?> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
            Function<Integer, OD2> d2, Function<Integer, OD3> d3) {
        return reshape(tensor,
                af.shape(d0.apply(tensor.shape().d0().intValue()), d1.apply(tensor.shape().d1().intValue()),
                        d2.apply(tensor.shape().d2().intValue()), d3.apply(tensor.shape().d3().intValue())));
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> reshape(
            Tensor<T, ?, ?, ?, ?> tensor, Shape<OD0, OD1, OD2, OD3> newShape) {
        assert tensor.shape().capacity() == newShape.capacity() : String.format(
                "New shape %s doesn't have same capacity as original shape %s", newShape, tensor.shape());
        return createFromOperation(tensor.type(), newShape,
                ptr -> arrayfire_h.af_moddims(ptr, tensor.dereference(), newShape.dims().length,
                        af.nativeDims(newShape)));
    }

    public static void release(Tensor<?, ?, ?, ?, ?> tensor) {
        handleStatus(arrayfire_h.af_release_array(tensor.dereference()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> eval(
            Tensor<T, D0, D1, D2, D3> tensor) {
        handleStatus(arrayfire_h.af_eval(tensor.dereference()));
        return tensor;
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> mul(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, ?, ?, ?, ?> rhs) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_mul(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> div(
            TensorLike<T, D0, D1, D2, D3> tensor, TensorLike<T, ?, ?, ?, ?> rhs) {
        return createFromOperation(tensor.tensor().type(), tensor.tensor().shape(),
                ptr -> arrayfire_h.af_div(ptr, tensor.tensor().dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> add(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, D0, D1, D2, D3> rhs) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_add(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sub(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, D0, D1, D2, D3> rhs) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sub(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<B8, D0, D1, D2, D3> gte(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, D0, D1, D2, D3> rhs) {
        return createFromOperation(DataType.B8, tensor.shape(),
                ptr -> arrayfire_h.af_ge(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, D0, D1, D2, D3> rhs) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_maxof(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor, TensorLike<T, D0, D1, D2, D3> rhs) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_minof(ptr, tensor.dereference(), rhs.tensor().dereference(), true));
    }


    public <T extends DataType<?, ?>, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, D1, D2, D3> join(
            Tensor<T, ?, D1, D2, D3> lhs, TensorLike<T, ?, D1, D2, D3> rhs) {
        return createFromOperation(lhs.type(),
                af.shape(af.n(lhs.shape().d0().intValue() + rhs.tensor().shape().d0().intValue()), lhs.shape().d1(),
                        lhs.shape().d2(), lhs.shape().d3()),
                ptr -> arrayfire_h.af_join(ptr, 0, lhs.dereference(), rhs.tensor().dereference()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, D2, D3> join(
            Tensor<T, D0, ?, D2, D3> lhs, TensorLike<T, D0, ?, D2, D3> rhs, arrayfire.dims.D1 ignored) {
        assert lhs.shape().d0().intValue() == rhs.tensor().shape().d0().intValue() && lhs.shape().d2().intValue() == rhs.tensor().shape().d2().intValue() && lhs.shape().d3().intValue() == rhs.tensor().shape().d3().intValue() : String.format(
                "Incompatible shapes to join along d1: %s, %s", lhs.shape(), rhs.tensor().shape());
        return createFromOperation(lhs.type(),
                af.shape(lhs.shape().d0(), af.n(lhs.shape().d1().intValue() + rhs.tensor().shape().d1().intValue()),
                        lhs.shape().d2(), lhs.shape().d3()),
                ptr -> arrayfire_h.af_join(ptr, 1, lhs.dereference(), rhs.tensor().dereference()));
    }


    public <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<ST, D0, U, D2, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.dims.D1 dim) {
        return reduce(tensor, arrayfire_h::af_sum, dim, tensor.type().sumType());
    }

    public <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<ST, U, D1, D2, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return reduce(tensor, arrayfire_h::af_sum, tensor.type().sumType());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return af.reduce(tensor, arrayfire_h::af_mean, d0, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.dims.D0 dim) {
        return af.reduce(tensor, arrayfire_h::af_mean, dim, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.dims.D1 dim) {
        return af.reduce(tensor, arrayfire_h::af_mean, dim, tensor.type());
    }


    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> median(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return af.reduce(tensor, arrayfire_h::af_median, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return af.reduce(tensor, arrayfire_h::af_max, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, U, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.dims.D1 dim) {
        return af.reduce(tensor, arrayfire_h::af_max, dim, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, U, D1, D2, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return af.reduce(tensor, arrayfire_h::af_min, tensor.type());
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> ImaxResult<T, U, D1, D2, D3> imax(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var shape = shape(af.u(), tensor.d1(), tensor.d2(), tensor.d3());
        var maxValues = new Tensor<>(tensor.type(), shape);
        var maxIndices = new Tensor<>(DataType.U32, shape);
        handleStatus(arrayfire_h.af_imax(maxValues.segment(), maxIndices.segment(), tensor.dereference(), 0));
        return new ImaxResult<>(maxValues, maxIndices);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, K extends Number> TopKResult<T, K, D1, D2, D3> topk(
            Tensor<T, D0, D1, D2, D3> tensor, K k) {
        var shape = shape(k, tensor.d1(), tensor.d2(), tensor.d3());
        var topValues = new Tensor<>(tensor.type(), shape);
        var topIndices = new Tensor<>(DataType.U32, shape);
        // TODO: Investigate fixed parameters.
        handleStatus(
                arrayfire_h.af_topk(topValues.segment(), topIndices.segment(), tensor.dereference(), k.intValue(), 0,
                        0));
        return new TopKResult<>(topValues, topIndices);

    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D0, D2, D3> diag(
            Tensor<T, D0, U, D2, D3> tensor) {
        // TODO: Investigate fixed parameters.
        return createFromOperation(tensor.type(),
                af.shape(tensor.shape().d0(), tensor.shape().d0(), tensor.shape().d2(), tensor.shape().d3()),
                ptr -> arrayfire_h.af_diag_create(ptr, tensor.dereference(), 0));
    }

    // https://arrayfire.org/docs/group__blas__func__matmul.htm
    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, OD1 extends Number> Tensor<T, D0, OD1, D2, D3> matmul(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D1, OD1, D2, D3> rhs) {
        assert tensor.shape().d1().intValue() == rhs.shape().d0().intValue() : String.format(
                "Misaligned shapes for matmul, left: %s right: %s", tensor.shape(), rhs.shape());
        // TODO: Investigate fixed parameters.
        return createFromOperation(tensor.type(),
                af.shape(tensor.shape().d0(), rhs.shape().d1(), tensor.shape().d2(), tensor.shape().d3()),
                ptr -> arrayfire_h.af_matmul(ptr, tensor.dereference(), rhs.dereference(), 0, 0));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> clamp(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, ?, ?> lo, Tensor<T, ?, ?, ?, ?> hi) {
        // TODO: Batch parameter.
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_clamp(ptr, tensor.dereference(), lo.dereference(), hi.dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> relu(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return clamp(tensor, af.constant(0f).cast(tensor.type()),
                af.constant(Float.POSITIVE_INFINITY).cast(tensor.type()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<B8, D0, D1, D2, D3> eq(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, ?, ?> rhs) {
        return createFromOperation(DataType.B8, tensor.shape(),
                ptr -> arrayfire_h.af_eq(ptr, tensor.dereference(), rhs.dereference(), true));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> negate(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var minusOne = af.constant(tensor.type(), tensor.shape(), -1);
        return mul(tensor, minusOne);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> exp(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return createFromOperation(tensor.type(), tensor.shape(), ptr -> arrayfire_h.af_exp(ptr, tensor.dereference()));

    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> log(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return createFromOperation(tensor.type(), tensor.shape(), ptr -> arrayfire_h.af_log(ptr, tensor.dereference()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> abs(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return createFromOperation(tensor.type(), tensor.shape(), ptr -> arrayfire_h.af_abs(ptr, tensor.dereference()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sqrt(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sqrt(ptr, tensor.dereference()));
    }

    public <T extends DataType<?, T>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> softmax(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var max = max(tensor);
        var normalized = sub(tensor, max.tileAs(tensor));
        var exp = normalized.exp();
        return div(exp.cast(exp.type().sumType()), sum(exp).tileAs(tensor.shape()));
    }

    public <T extends DataType<?, T>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> softmax(
            Tensor<T, D0, D1, D2, D3> tensor, float temperature) {
        var max = max(tensor);
        var normalized = sub(tensor, max.tileAs(tensor));
        var exp = exp(div(normalized, constant(tensor.type(), tensor.shape(), temperature)));
        return div(exp, sum(exp).tileAs(tensor));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sigmoid(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var one = af.ones(tensor);
        return div(one, add(one, exp(negate(tensor))));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> sparse(
            Tensor<T, D0, D1, D2, D3> tensor, Storage storage) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_create_sparse_array_from_dense(ptr, tensor.dereference(), storage.code()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, U, U, U> index(
            Tensor<T, D0, D1, D2, D3> tensor, Index index) {
        return (Tensor<T, N, U, U, U>) index(tensor, new Index[]{index});
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, N, U, U> index(
            Tensor<T, D0, D1, D2, D3> tensor, Index i0, Index i1) {
        return (Tensor<T, N, N, U, U>) index(tensor, new Index[]{i0, i1});
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, U, U> index(
            Tensor<T, D0, D1, D2, D3> tensor, Span span, Index i1) {
        return (Tensor<T, D0, N, U, U>) index(tensor, new Index[]{af.seq(tensor.shape().d0()), i1});
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, N, U> index(
            Tensor<T, D0, D1, D2, D3> tensor, Span span0, Span span1, Index i2) {
        return (Tensor<T, D0, D1, N, U>) index(tensor,
                new Index[]{af.seq(tensor.shape().d0()), af.seq(tensor.shape().d1()), i2});
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, U, U> index(
            Tensor<T, D0, D1, D2, D3> tensor, Index i0, Span span) {
        return (Tensor<T, D0, N, U, U>) index(tensor, new Index[]{i0, af.seq(tensor.shape().d1())});
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, ?, ?, ?, ?> index(
            Tensor<T, D0, D1, D2, D3> tensor, Index... indexes) {
        try (Arena arena = Arena.ofConfined()) {
            var layout = MemoryLayout.sequenceLayout(indexes.length, Index.LAYOUT);
            var nativeIndexes = arena.allocateArray(Index.LAYOUT, indexes.length);
            for (int i = 0; i < indexes.length; i++) {
                indexes[i].emigrate(
                        nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(i)),
                                Index.LAYOUT.byteSize()));
            }
            var result = arena.allocate(ValueLayout.ADDRESS);
            handleStatus(arrayfire_h.af_index_gen(result, tensor.dereference(), indexes.length, nativeIndexes));

            // We don't obviously know the new shape, so we need to compute it.
            var dims = arena.allocateArray(ValueLayout.JAVA_LONG, 4);
            handleStatus(arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                    result.get(ValueLayout.ADDRESS, 0)));
            var d0 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 0);
            var d1 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 1);
            var d2 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 2);
            var d3 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 3);
            var resultTensor = new Tensor<>(tensor.type(), af.shape(af.n(d0), af.n(d1), af.n(d2), af.n(d3)));
            resultTensor.segment().copyFrom(result);
            return resultTensor;
        }
    }


    // zip two tensors together
    public <LT extends DataType<?, ?>, RT extends DataType<?, ?>, LD0 extends Number, RD0 extends Number, D1 extends Number> ZipD1<LT, RT, LD0, RD0, D1> zip(
            Tensor<LT, LD0, D1, U, U> left, Tensor<RT, RD0, D1, U, U> right) {
        return new ZipD1<>(left, right);
    }

    public <LT extends DataType<?, ?>, RT extends DataType<?, ?>, LD0 extends Number, RD0 extends Number> List<ZipD1<LT, RT, LD0, RD0, N>> batch(
            ZipD1<LT, RT, LD0, RD0, ?> zip, int batchSize) {
        return batch(zip, d1, batchSize);
    }

    public <LT extends DataType<?, ?>, RT extends DataType<?, ?>, LD0 extends Number, RD0 extends Number> List<ZipD1<LT, RT, LD0, RD0, N>> batch(
            ZipD1<LT, RT, LD0, RD0, ?> zip, arrayfire.dims.D1 ignored, int batchSize) {
        var left = batch(zip.left(), IntNumber::n, batchSize);
        var right = batch(zip.right(), IntNumber::n, batchSize);
        return IntStream.range(0, left.size()).mapToObj(i -> af.zip(left.get(i), right.get(i))).toList();
    }


    // unbatch
    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, N, D2, D3> unbatch(
            List<Tensor<T, D0, N, D2, D3>> tensors, arrayfire.dims.D1 ignored) {
        return tensors.stream().reduce((a, b) -> {
            // TODO: We could do this quicker if we use the variable length join method (up to 10).
            var joined = af.join(a, b, d1);
            a.release();
            b.release();
            return joined;
        }).orElseThrow();
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> List<Tensor<T, D0, N, U, U>> batch(
            Tensor<T, D0, D1, U, U> tensor, int batchSize) {
        return batch(tensor, af::n, batchSize);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, BDT extends Number> List<Tensor<T, D0, BDT, U, U>> batch(
            Tensor<T, D0, D1, D2, D3> tensor, Function<Integer, BDT> type, int batchSize) {
        var results = new ArrayList<Tensor<T, D0, BDT, U, U>>();
        var d0Seq = af.seq(0, tensor.shape().d0().intValue() - 1);
        for (int i = 0; i < tensor.shape().d1().intValue(); i += batchSize) {
            var computedD1Size = Math.min(batchSize, tensor.shape().d1().intValue() - i);
            var slice = index(tensor, d0Seq, af.seq(i, i + computedD1Size - 1));
            results.add(slice.reshape(shape(tensor.shape().d0(), type.apply(computedD1Size))));
        }
        return results;
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
            Tensor<T, ?, ?, ?, ?> tensor, TensorLike<T, OD0, OD1, OD2, OD3> newShapeTensor) {
        return tileAs(tensor, newShapeTensor.tensor().shape());
    }

    public <T extends DataType<?, ?>, OD0 extends Number, OD1 extends Number, OD2 extends Number, OD3 extends Number> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
            Tensor<T, ?, ?, ?, ?> tensor, Shape<OD0, OD1, OD2, OD3> newShape) {
        assert newShape.capacity() % tensor.shape().capacity() == 0 : String.format(
                "Can't tile perfectly from %s to %s", tensor.shape(), newShape);
        int d0ratio = newShape.d0().intValue() / tensor.shape().d0().intValue();
        int d1ratio = newShape.d1().intValue() / tensor.shape().d1().intValue();
        int d2ratio = newShape.d2().intValue() / tensor.shape().d2().intValue();
        int d3ratio = newShape.d3().intValue() / tensor.shape().d3().intValue();
        return createFromOperation(tensor.type(), newShape,
                ptr -> arrayfire_h.af_tile(ptr, tensor.dereference(), d0ratio, d1ratio, d2ratio, d3ratio));
    }

    public <T extends DataType<?, ?>> Tensor<T, N, U, U, U> flatten(Tensor<T, ?, ?, ?, ?> tensor) {
        return reshape(tensor, shape(tensor.shape().capacity()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, D3, U, U> flatten3(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return reshape(tensor,
                shape(tensor.shape().d0().intValue() * tensor.shape().d1().intValue() * tensor.shape().d2().intValue(),
                        tensor.shape().d3()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, N, N, U, U> flatten2(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return reshape(tensor, shape(tensor.shape().d0().intValue() * tensor.shape().d1().intValue(),
                tensor.shape().d2().intValue() * tensor.shape().d3().intValue()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> flip(
            Tensor<T, D0, D1, D2, D3> tensor) {
        // TODO: Investigate fixed parameters.
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_flip(ptr, tensor.dereference(), 0));
    }


    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters) {
        return convolve2(tensor, filters, af.shape(1, 1), af.shape(0, 0), af.shape(1, 1));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride) {
        return convolve2(tensor, filters, stride, af.shape(0, 0), af.shape(1, 1));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride,
            Shape<?, ?, ?, ?> padding) {
        return convolve2(tensor, filters, stride, padding, af.shape(1, 1));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number, FD3 extends Number> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride,
            Shape<?, ?, ?, ?> padding, Shape<?, ?, ?, ?> dilation) {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.ADDRESS);
            // TODO(https://github.com/arrayfire/arrayfire/issues/3402): Convolutions look like they may allocate memory outside of ArrayFire's scope, so sometimes we need to GC first.
            retryWithGc(() -> handleStatus(
                    arrayfire_h.af_convolve2_nn(result, tensor.dereference(), filters.dereference(), 2,
                            af.nativeDims(stride), 2, af.nativeDims(padding), 2, af.nativeDims(dilation))));
            var computedDims = af.getDims(result);
            var resultTensor = new Tensor<>(tensor.type(),
                    shape(n((int) computedDims[0]), n((int) computedDims[1]), filters.shape().d3(),
                            tensor.shape().d3()));
            resultTensor.segment().copyFrom(result);
            return resultTensor;
        }
    }

    /**
     * L2 norm.
     */
    public <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<ST, U, D1, D2, D3> norm(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var mul = mul(tensor, tensor);
        var sum = sum(mul);
        return sqrt(sum);
    }

    /**
     * Normalize by dividing by the L2 norm.
     */
    public <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<ST, D0, D1, D2, D3> normalize(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return div(cast(tensor, tensor.type().sumType()), norm(tensor).tileAs(tensor.shape()));
    }

    /**
     * Center by subtracting the average.
     */
    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number> Tensor<T, D0, D1, D2, D3> center(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return sub(tensor, mean(tensor).tileAs(tensor));
    }

    // svd
    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number> SvdResult<T, D0, D1> svd(
            Tensor<T, D0, D1, U, U> tensor) {
        var u = new Tensor<>(tensor.type(), shape(tensor.shape().d1(), tensor.shape().d1()));
        var s = new Tensor<>(tensor.type(), shape(tensor.shape().d1()));
        var v = new Tensor<>(tensor.type(), shape(tensor.shape().d0(), tensor.shape().d0()));
        handleStatus(arrayfire_h.af_svd(u.segment(), s.segment(), v.segment(), tensor.dereference()));
        return new SvdResult<>(u, s, v);
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D0, U, U> cov(
            Tensor<T, D0, D1, U, U> tensor) {
        var subMean = sub(tensor, mean(tensor, af.d1).tileAs(tensor));
        var matrix = matmul(subMean, subMean.transpose());
        return div(matrix, af.constant(matrix.type(), matrix.shape(), tensor.shape().d1().floatValue() - 1.0f));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D0, U, U> zcaMatrix(
            Tensor<T, D0, D1, U, U> tensor) {
        return tidy(() -> {
            var cov = cov(tensor);
            var svd = svd(cov);
            var invSqrtS = diag(div(af.constant(svd.s().type(), svd.s().shape(), 1.0f),
                    sqrt(af.add(svd.s(), af.constant(svd.s().type(), svd.s().shape(), 1e-5f)))));
            return matmul(svd.u(), matmul(invSqrtS, svd.u().transpose()));
        });
    }


    public <T extends DataType<?, ?>, D extends Number> Tensor<T, D, D, U, U> inverse(
            TensorLike<T, D, D, U, U> tensor) {
        return createFromOperation(tensor.tensor().type(), tensor.tensor().shape(),
                ptr -> arrayfire_h.af_inverse(ptr, tensor.tensor().dereference(), 0));
    }

    // TODO: Add uncropped version.
    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number> Tensor<T, D0, D1, U, U> rotate(
            Tensor<T, D0, D1, U, U> tensor, float angle, InterpolationType interpolationType) {
        return createFromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_rotate(ptr, tensor.dereference(), angle, true, interpolationType.code()));
    }

    public <T extends DataType<?, ?>, D0 extends Number, D1 extends Number, ND0 extends Number, ND1 extends Number> Tensor<T, ND0, ND1, U, U> scale(
            Tensor<T, D0, D1, U, U> tensor, ND0 nd0, ND1 nd1, InterpolationType interpolationType) {
        return createFromOperation(tensor.type(), shape(nd0, nd1, tensor.shape().d2(), tensor.shape().d3()),
                ptr -> arrayfire_h.af_scale(ptr, tensor.dereference(), nd0.floatValue() / tensor.d0().floatValue(),
                        nd1.floatValue() / tensor.d1().floatValue(), nd0.longValue(), nd1.longValue(),
                        interpolationType.code()));
    }

    public static record DeviceMemInfo(long allocBytes, long allocBuffers, long lockBytes, long lockBuffers) {
    }

    public DeviceMemInfo deviceMemInfo() {
        var allocator = af.allocator();
        var allocBytes = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
        var allocBuffers = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
        var lockBytes = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
        var lockBuffers = allocator.allocateArray(ValueLayout.JAVA_LONG, 1);
        handleStatus(arrayfire_h.af_device_mem_info(allocBytes, allocBuffers, lockBytes, lockBuffers));
        return new DeviceMemInfo(allocBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
                allocBuffers.getAtIndex(ValueLayout.JAVA_LONG, 0), lockBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
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
            if (e.status() == Status.AF_ERR_NO_MEM) {
                deviceGc();
                fn.run();
            } else {
                throw e;
            }
        }
    }

    private long[] nativeDims(Tensor<?, ?, ?, ?, ?> tensor) {
        var dims = af.allocator().allocateArray(ValueLayout.JAVA_LONG, 4);
        handleStatus(arrayfire_h.af_get_dims(dims, dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                tensor.dereference()));
        return dims.toArray(ValueLayout.JAVA_LONG);
    }

    private Shape<?, ?, ?, ?> nativeShape(Tensor<?, ?, ?, ?, ?> tensor) {
        var dims = nativeDims(tensor);
        return af.shape((int) dims[0], (int) dims[1], (int) dims[2], (int) dims[3]);
    }

    static void handleStatus(Object res) {
        var result = Status.fromCode((int) res);
        if (!Status.AF_SUCCESS.equals(result)) {
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
