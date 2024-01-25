package arrayfire;

import arrayfire.autograd.GradFunction;
import arrayfire.capi.arrayfire_h;
import arrayfire.containers.NativeArray;
import arrayfire.numbers.*;
import arrayfire.optimizers.OptimizerProvider;
import arrayfire.utils.Functions;
import arrayfire.utils.Reference;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;

public class ArrayFire {

    public static final U8 U8 = new U8();
    public static final U64 U64 = new U64();
    public static final U32 U32 = new U32();
    public static final F32 F32 = new F32();
    public static final F16 F16 = new F16();
    public static final F64 F64 = new F64();
    public static final B8 B8 = new B8();
    public static final S32 S32 = new S32();

    public static final arrayfire.D0 D0 = new D0();
    public static final arrayfire.D1 D1 = new D1();
    public static final arrayfire.D2 D2 = new D2();
    public static final arrayfire.D3 D3 = new D3();

    private static boolean successfullyLoadedLibraries = false;

    private static void maybeLoadNativeLibraries() {
        if (successfullyLoadedLibraries) {
            return;
        }
        var libraries = List.of("af", "afcuda", "afopencl", "afcpu");
        Throwable firstThrowable = null;
        for (var library : libraries) {
            try {
                System.loadLibrary(library);
                successfullyLoadedLibraries = true;
                tidy(() -> {
                    var version = version();
                    if (version.major() < 3 || (version.major() == 3 && version.minor() < 8)) {
                        throw new IllegalStateException(
                                String.format("Unsupported ArrayFire version, should be >= 3.8.0: %s", version));
                    }
                });
                return;
            } catch (Throwable throwable) {
                if (firstThrowable == null) {
                    firstThrowable = throwable;
                }
            }
        }
        throw new RuntimeException(
                "Failed to load ArrayFire native libraries, make sure it is installed at the required version.",
                firstThrowable);
    }

    /**
     * Executes the given function in a new memory scope, and disposes of all memory allocated in that scope afterward.
     */
    public static void tidy(Runnable fn) {
        Scope.tidy(fn);
    }

    /**
     * Executes the given function in a new scope, and disposes of all memory allocated in that scope except the value returned by the function if it is manually managed memory container.
     */
    public static <T> T tidy(Supplier<T> fn) {
        var parentScope = scope();
        var resultReference = new Reference<T>();
        tidy(() -> {
            var result = (T) fn.get();
            if (result instanceof MemoryContainer mc) {
                MemoryScope.move(mc, parentScope.memory());
            }
            resultReference.set(result);
        });
        return resultReference.get();
    }

    /**
     * Returns the current scope of the thread.
     */
    public static Scope scope() {
        return Scope.current();
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor) {
        return sort(tensor, D0);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sort(tensor, dim, true);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sort(ptr, tensor.dereference(), dim.index(), ascending)).withName(
                "sort").withInput(tensor).withoutGradFunction();
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor) {
        return sortIndex(tensor, D0);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sortIndex(tensor, dim, true);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
            Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        try (Arena arena = Arena.ofConfined()) {
            var values = arena.allocate(Tensor.LAYOUT);
            var indices = arena.allocate(Tensor.LAYOUT);
            handleStatus(
                    () -> arrayfire_h.af_sort_index(values, indices, tensor.dereference(), dim.index(), ascending));
            return new SortIndexResult<>(
                    fromSegment(tensor.type(), tensor.shape(), values).withName("sort_values").withInput(
                            tensor).withoutGradFunction(),
                    fromSegment(U32, tensor.shape(), indices).withName("sort_indices").withInput(
                            tensor).withoutGradFunction());
        }
    }

    /**
     * Create a random permutation of indices for the given dimension.
     */
    public static <D extends Num<D>> Index<D> permutation(D dim) {
        var indices = tidy(() -> sortIndex(randu(U32, shape(dim))).indices());
        return af.seq(indices);
    }

    public static <DT extends DataType<?, ?>, AT extends NativeArray<DT, ?, ?>> Tensor<DT, N, U, U, U> create(
            AT array) {
        return create(array, shape(n(array.length())));
    }

    public static <DT extends DataType<?, ?>, AT extends NativeArray<DT, ?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> create(
            AT array, Shape<D0, D1, D2, D3> shape) {
        try (Arena arena = Arena.ofConfined()) {
            return fromOperation(array.type(), shape,
                    ptr -> arrayfire_h.af_create_array(ptr, array.segment(), shape.dims().length,
                            nativeDims(arena, shape), array.type().code())).withName("create").withoutInputs();
        }
    }


    @SafeVarargs
    public static <JT, AT extends NativeArray<DT, JT, ?>, DT extends DataType<AT, ?>> Tensor<DT, N, U, U, U> create(
            DT type, JT... values) {
        return tidy(() -> {
            var array = type.create(values.length);
            for (int i = 0; i < values.length; i++) {
                array.set(i, values[i]);
            }
            return create(array);
        });
    }

    @SuppressWarnings("unchecked")
    public static <JT, JTA, AT extends NativeArray<DT, JT, JTA>, DT extends DataType<AT, ?>> Tensor<DT, N, U, U, U> create(
            DT type, JTA values) {
        return tidy(() -> {
            var length = Array.getLength(values);
            var array = type.create(length);
            for (int i = 0; i < length; i++) {
                array.set(i, (JT) Array.get(values, i));
            }
            return create(array);
        });
    }

    public static Tensor<F32, N, U, U, U> create(float... values) {
        return create(F32, values);
    }

    public static Tensor<F64, N, U, U, U> create(double... values) {
        return create(F64, values);
    }

    public static Tensor<S32, N, U, U, U> create(int... values) {
        return create(S32, values);
    }

    public static Tensor<F32, U, U, U, U> constant(float value) {
        return constant(F32, value);
    }

    public static <DT extends DataType<?, ?>> Tensor<DT, U, U, U, U> constant(DT type, double value) {
        return constant(type, shape(u()), value);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> constant(
            DT type, Shape<D0, D1, D2, D3> shape, double value) {
        try (Arena arena = Arena.ofConfined()) {
            return fromOperation(type, shape,
                    ptr -> arrayfire_h.af_constant(ptr, value, shape.dims().length, nativeDims(arena, shape),
                            type.code())).withName("constant").withoutInputs();
        }
    }

    public static void sync() {
        handleStatus(() -> arrayfire_h.af_sync(deviceId()));
    }

    public static Index<N> seq(int begin, int endInclusive, int step) {
        return new Index<>(new Seq(begin, endInclusive, step), ArrayFire::n);
    }

    public static Index<N> seq(int begin, int endInclusive) {
        return new Index<>(new Seq(begin, endInclusive, 1), ArrayFire::n);
    }

    public static <DT extends DataType<?, ?>, D0 extends Num<D0>> Index<D0> seq(Tensor<DT, D0, U, U, U> index) {
        return new Index<>(index, index.d0()::create);
    }

    public static <D extends Num<D>> Index<D> seq(D num) {
        return new Index<>(new Seq(0, num.size() - 1, 1), num::create);
    }

    public static <D extends Num<D>> Index<D> seq(int offset, D num) {
        return new Index<>(new Seq(offset, num.size() - 1 + offset, 1), num::create);
    }

    public static Span span() {
        return new Span();
    }

    public static Shape<N, U, U, U> shape(int d0) {
        return new Shape<>(n(d0), u(), u(), u());
    }

    public static <D0 extends Num<?>> Shape<D0, U, U, U> shape(D0 d0) {
        return new Shape<>(d0, u(), u(), u());
    }

    public static <D0 extends Num<?>> Shape<D0, N, U, U> shape(D0 d0, int d1) {
        return new Shape<>(d0, n(d1), u(), u());
    }

    public static <D1 extends Num<?>> Shape<N, D1, U, U> shape(int d0, D1 d1) {
        return new Shape<>(n(d0), d1, u(), u());
    }

    public static Shape<N, N, U, U> shape(int d0, int d1) {
        return new Shape<>(n(d0), n(d1), u(), u());
    }

    public static <D0 extends Num<?>, D1 extends Num<?>> Shape<D0, D1, U, U> shape(D0 d0, D1 d1) {
        return new Shape<>(d0, d1, u(), u());
    }

    public static Shape<N, N, N, U> shape(int d0, int d1, int d2) {
        return new Shape<>(n(d0), n(d1), n(d2), u());
    }


    public static Shape<N, N, N, N> shape(int d0, int d1, int d2, int d3) {
        return new Shape<>(n(d0), n(d1), n(d2), n(d3));
    }

    public static <D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>> Shape<D0, D1, D2, U> shape(D0 d0, D1 d1,
                                                                                                       D2 d2) {
        return new Shape<>(d0, d1, d2, u());
    }

    public static <D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Shape<D0, D1, D2, D3> shape(
            D0 d0, D1 d1, D2 d2, D3 d3) {
        return new Shape<>(d0, d1, d2, d3);
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, U, D1, D2, D3>.Unary<IT, D0, D1, D2, D3> reduce(
            String name, Tensor<IT, D0, D1, D2, D3> a,
            Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, T resultType) {
        return reduce(name, a, method, D0, resultType);
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, U, D1, D2, D3>.Unary<IT, D0, D1, D2, D3> reduce(
            String name, Tensor<IT, D0, D1, D2, D3> a,
            Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D0 dim,
            T resultType) {
        return fromOperation(resultType, shape(u(), a.d1(), a.d2(), a.d3()),
                ptr -> method.apply(ptr, a.dereference(), dim.index())).withName(name).withInput(a);
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, U, D2, D3>.Unary<IT, D0, D1, D2, D3> reduce(
            String name, Tensor<IT, D0, D1, D2, D3> a,
            Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D1 dim,
            T resultType) {
        return fromOperation(resultType, shape(a.d0(), u(), a.d2(), a.d3()),
                ptr -> method.apply(ptr, a.dereference(), dim.index())).withName(name).withInput(a);
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, D1, U, D3>.Unary<IT, D0, D1, D2, D3> reduce(
            String name, Tensor<IT, D0, D1, D2, D3> a,
            Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D2 dim,
            T resultType) {
        return fromOperation(resultType, shape(a.d0(), a.d1(), u(), a.d3()),
                ptr -> method.apply(ptr, a.dereference(), dim.index())).withName(name).withInput(a);
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, D1, D2, U>.Unary<IT, D0, D1, D2, D3> reduce(
            String name, Tensor<IT, D0, D1, D2, D3> a,
            Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D3 dim,
            T resultType) {
        return fromOperation(resultType, shape(a.d0(), a.d1(), a.d2(), u()),
                ptr -> method.apply(ptr, a.dereference(), dim.index())).withName(name).withInput(a);
    }


    @SuppressWarnings("unchecked")
    public static <T extends DataType<?, ?>, OT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<OT, D0, D1, D2, D3> cast(
            Tensor<T, D0, D1, D2, D3> input, OT type) {
        if (input.type().equals(type)) {
            return (Tensor<OT, D0, D1, D2, D3>) input;
        }
        return fromOperation(type, input.shape(),
                ptr -> arrayfire_h.af_cast(ptr, input.dereference(), type.code())).withName("cast").withInput(
                input).withGradFunction((result, grads) -> grads.cast(input.type()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> ones(
            Tensor<T, D0, D1, D2, D3> model) {
        return ones(model.type(), model.shape());
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> ones(
            T type, Shape<D0, D1, D2, D3> shape) {
        return constant(type, 1).tileAs(shape);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> zeros(
            T type, Shape<D0, D1, D2, D3> shape) {
        return constant(type, 0).tileAs(shape);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> randu(
            T type, Shape<D0, D1, D2, D3> shape) {
        try (Arena arena = Arena.ofConfined()) {
            return fromOperation(type, shape,
                    ptr -> arrayfire_h.af_randu(ptr, shape.dims().length, nativeDims(arena, shape),
                            type.code())).withName("randu").withoutInputs();
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> randn(
            T type, Shape<D0, D1, D2, D3> shape) {
        try (Arena arena = Arena.ofConfined()) {
            return fromOperation(type, shape,
                    ptr -> arrayfire_h.af_randn(ptr, shape.dims().length, nativeDims(arena, shape),
                            type.code())).withName("randn").withoutInputs();
        }
    }

    public static Tensor<U32, N, U, U, U> range(int n) {
        return range(U32, n);
    }

    public static <T extends DataType<?, ?>> Tensor<T, N, U, U, U> range(T type, int n) {
        try (Arena arena = Arena.ofConfined()) {
            var shape = shape(n(n));
            return fromOperation(type, shape,
                    ptr -> arrayfire_h.af_range(ptr, shape.dims().length, nativeDims(arena, shape), 0,
                            type.code())).withName("range").withoutInputs();
        }
    }

    public static void setSeed(long seed) {
        handleStatus(() -> arrayfire_h.af_set_seed(seed));
    }

    public static void setRandomEngineType(RandomEngineType type) {
        handleStatus(() -> arrayfire_h.af_set_default_random_engine_type(type.code()));
    }

    public static <AT extends NativeArray<?, ?, ?>, T extends DataType<AT, ?>> AT data(Tensor<T, ?, ?, ?, ?> a) {
        var result = a.type().create(a.capacity());
        handleStatus(() -> arrayfire_h.af_get_data_ptr(result.segment(), a.dereference()));
        return result;
    }

    public static void checkDims(Tensor<?, ?, ?, ?, ?> tensor) {
        var trueDims = getDims(tensor.segment());
        var expectedDims = tensor.shape().dims();
        for (int i = 0; i < trueDims.length; i++) {
            if (trueDims[i] != expectedDims[i]) {
                throw new IllegalStateException(
                        String.format("Expected dimensions %s but got %s", Arrays.toString(expectedDims),
                                Arrays.toString(trueDims)));
            }
        }
    }

    public static long[] getDims(MemorySegment a) {
        try (Arena arena = Arena.ofConfined()) {
            var dims = arena.allocateArray(ValueLayout.JAVA_LONG, 4);
            handleStatus(
                    () -> arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                            a.getAtIndex(ValueLayout.ADDRESS, 0)));
            return dims.toArray(ValueLayout.JAVA_LONG);
        }
    }

    public static Version version() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocateArray(ValueLayout.JAVA_INT, 3);
            handleStatus(() -> arrayfire_h.af_get_version(result, result.asSlice(4), result.asSlice(8)));
            var arr = result.toArray(ValueLayout.JAVA_INT);
            return new Version(arr[0], arr[1], arr[2]);
        }
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

    public static Set<Backend> availableBackends() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_available_backends(result));
            return Backend.fromBitmask(result.get(ValueLayout.JAVA_INT, 0));
        }
    }

    public static Backend backend() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_active_backend(result));
            return Backend.fromCode(result.get(ValueLayout.JAVA_INT, 0));
        }
    }

    public static void setBackend(Backend backend) {
        handleStatus(() -> arrayfire_h.af_set_backend(backend.code()));
    }

    public static int deviceId() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_device(result));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    public static void setDeviceId(int device) {
        handleStatus(() -> arrayfire_h.af_set_device(device));
    }

    public static int deviceCount() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_device_count(result));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    public static DeviceInfo deviceInfo() {
        try (Arena arena = Arena.ofConfined()) {
            var name = arena.allocateArray(ValueLayout.JAVA_CHAR, 64);
            var platform = arena.allocateArray(ValueLayout.JAVA_CHAR, 64);
            var toolkit = arena.allocateArray(ValueLayout.JAVA_CHAR, 64);
            var compute = arena.allocateArray(ValueLayout.JAVA_CHAR, 64);
            handleStatus(() -> arrayfire_h.af_device_info(name, platform, toolkit, compute));
            return new DeviceInfo(name.getUtf8String(0), platform.getUtf8String(0), toolkit.getUtf8String(0),
                    compute.getUtf8String(0));
        }
    }

    private static MemorySegment nativeDims(Arena arena, Shape<?, ?, ?, ?> shape) {
        return arena.allocateArray(ValueLayout.JAVA_LONG, shape.dims());
    }


    private static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, D1, D2, D3> fromSegment(
            T type, Shape<D0, D1, D2, D3> shape, MemorySegment segment) {
        var tb = new TensorBuilder<T, D0, D1, D2, D3>();
        tb.tensor = new Tensor<>(type, shape);
        tb.tensor.segment().copyFrom(segment);
        return tb;
    }

    private static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, D1, D2, D3> fromOperation(
            T type, Shape<D0, D1, D2, D3> shape, Function<MemorySegment, Integer> method) {
        try (Arena arena = Arena.ofConfined()) {
            var tb = new TensorBuilder<T, D0, D1, D2, D3>();
            var segment = arena.allocate(Tensor.LAYOUT);
            handleStatus(() -> method.apply(segment));
            tb.tensor = new Tensor<>(type, shape);
            tb.tensor.segment().copyFrom(segment);
            return tb;
        }
    }

    private static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> TensorBuilder<T, D0, D1, D2, D3> fromTidy(
            Supplier<Tensor<T, D0, D1, D2, D3>> supplier) {
        var tb = new TensorBuilder<T, D0, D1, D2, D3>();
        tb.tensor = af.tidy(supplier);
        return tb;
    }

    public static class TensorBuilder<T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> {
        String name = "unknown";
        List<Tensor<?, ?, ?, ?, ?>> inputs = new ArrayList<>();
        Tensor<T, D0, D1, D2, D3> tensor;

        public TensorBuilder<T, D0, D1, D2, D3> withName(String name) {
            this.name = name;
            return this;
        }

        public <I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>> Unary<I0T, I0D0, I0D1, I0D2, I0D3> withInput(
                Tensor<I0T, I0D0, I0D1, I0D2, I0D3> input) {
            inputs.add(input);
            return new Unary<>();
        }

        public <I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>> Binary<I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> withInputs(
                Tensor<I0T, I0D0, I0D1, I0D2, I0D3> input0, Tensor<I1T, I1D0, I1D1, I1D2, I1D3> input1) {
            inputs.addAll(List.of(input0, input1));
            return new Binary<>();
        }

        public Tensor<T, D0, D1, D2, D3> withoutInputs() {
            scope().graph().add(new Graph.Node(name, tensor, List.of(), (grads) -> List.of()));
            return tensor;
        }

        public class Unary<I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>> {

            @SuppressWarnings("unchecked")
            public Tensor<T, D0, D1, D2, D3> withGradFunction(
                    GradFunction.Unary<T, D0, D1, D2, D3, I0T, I0D0, I0D1, I0D2, I0D3> unaryGradFunction) {
                // Register any inputs that are params.
                inputs.stream().filter(tl -> (tl instanceof Params)).forEach(input -> {
                    scope().graph().addParams((Params<?, ?, ?, ?, ?>) input);
                });
                scope().graph().add(
                        new Graph.Node(name, tensor, inputs, (grads) -> {
                            var inputGrad = unaryGradFunction.grads(tensor, (Tensor<T, D0, D1, D2, D3>) grads);
                            return List.of(inputGrad);
                        }));
                return tensor;
            }

            public Tensor<T, D0, D1, D2, D3> withoutGradFunction() {
                inputs.stream().filter(tl -> (tl instanceof Params)).forEach(input -> {
                    scope().graph().addParams((Params<?, ?, ?, ?, ?>) input);
                });
                scope().graph().add(
                        new Graph.Node(name, tensor, inputs, null));
                return tensor;
            }
        }

        public class Binary<I0T extends DataType<?, ?>, I0D0 extends Num<?>, I0D1 extends Num<?>, I0D2 extends Num<?>, I0D3 extends Num<?>, I1T extends DataType<?, ?>, I1D0 extends Num<?>, I1D1 extends Num<?>, I1D2 extends Num<?>, I1D3 extends Num<?>> {

            @SuppressWarnings("unchecked")
            public Tensor<T, D0, D1, D2, D3> withGradFunction(
                    GradFunction.Binary<T, D0, D1, D2, D3, I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> binaryGradFunction) {
                inputs.stream().filter(tl -> (tl instanceof Params)).forEach(input -> {
                    scope().graph().addParams((Params<?, ?, ?, ?, ?>) input);
                });
                scope().graph().add(
                        new Graph.Node(name, tensor, inputs, (grads) -> {
                            var inputGrads = binaryGradFunction.grads(tensor, (Tensor<T, D0, D1, D2, D3>) grads);
                            return List.of(inputGrads.left(), inputGrads.right());
                        }));
                return tensor;
            }

            public Tensor<T, D0, D1, D2, D3> withoutGradFunction() {
                inputs.stream().filter(tl -> (tl instanceof Params)).forEach(input -> {
                    scope().graph().addParams((Params<?, ?, ?, ?, ?>) input);
                });
                scope().graph().add(
                        new Graph.Node(name, tensor, inputs, null));
                return tensor;
            }
        }

//        public class Trinary<I0T extends DataType<?, ?>, I0D0 extends IntNumber<?>, I0D1 extends IntNumber<?>, I0D2 extends IntNumber<?>, I0D3 extends IntNumber<?>, I1T extends DataType<?, ?>, I1D0 extends IntNumber<?>, I1D1 extends IntNumber<?>, I1D2 extends IntNumber<?>, I1D3 extends IntNumber<?>, I2T extends DataType<?, ?>, I2D0 extends IntNumber<?>, I2D1 extends IntNumber<?>, I2D2 extends IntNumber<?>, I2D3 extends IntNumber<?>> {
//
//            @SuppressWarnings("unchecked")
//            public Tensor<T, D0, D1, D2, D3> withGradFunction(
//                    GradFunction.Binary<T, D0, D1, D2, D3, I0T, I0D0, I0D1, I0D2, I0D3, I1T, I1D0, I1D1, I1D2, I1D3> binaryGradFunction) {
//                memoryScope().graph().add(new Graph.Node(name, tensor, inputs, (grads) -> {
//                    var inputGrads = binaryGradFunction.grads(tensor, (Tensor<T, D0, D1, D2, D3>) grads);
//                    return List.of(inputGrads.left(), inputGrads.right());
//                }));
//                return tensor;
//            }
//
//            public Tensor<T, D0, D1, D2, D3> withoutGradFunction() {
//                memoryScope().graph().add(new Graph.Node(name, tensor, inputs, null));
//                return tensor;
//            }
//        }
    }


    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D1, D0, D2, D3> transpose(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(),
                shape(tensor.d1(), tensor.d0(), tensor.d2(),
                        tensor.d3()),
                ptr -> arrayfire_h.af_transpose(ptr, tensor.dereference(), false)).withName(
                "transpose").withInput(tensor).withGradFunction((result, grads) -> transpose(grads));

    }

    public static <T extends DataType<?, ?>, OD0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, D1, D2, D3> castshape(
            Tensor<T, ?, D1, D2, D3> tensor, Function<Integer, OD0> d0) {
        return reshape(tensor, shape(d0.apply(tensor.d0().size()), tensor.d1(), tensor.d2(),
                tensor.d3()));
    }

    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, OD1, D2, D3> castshape(
            Tensor<T, ?, ?, D2, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1) {
        return reshape(tensor,
                shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()), tensor.d2(),
                        tensor.d3()));
    }

    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, OD1, OD2, D3> castshape(
            Tensor<T, ?, ?, ?, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
            Function<Integer, OD2> d2) {
        return reshape(tensor, shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()),
                d2.apply(tensor.d2().size()), tensor.d3()));
    }

    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<T, OD0, OD1, OD2, OD3> castshape(
            Tensor<T, ?, ?, ?, ?> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
            Function<Integer, OD2> d2, Function<Integer, OD3> d3) {
        return reshape(tensor, shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()),
                d2.apply(tensor.d2().size()), d3.apply(tensor.d3().size())));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<T, OD0, OD1, OD2, OD3> reshape(
            Tensor<T, D0, D1, D2, D3> tensor, Shape<OD0, OD1, OD2, OD3> newShape) {
        assert tensor.shape().capacity() == newShape.capacity() : String.format(
                "New shape %s doesn't have same capacity as original shape %s", newShape, tensor.shape());
        try (Arena arena = Arena.ofConfined()) {
            return fromOperation(tensor.type(), newShape,
                    ptr -> arrayfire_h.af_moddims(ptr, tensor.dereference(), newShape.dims().length,
                            nativeDims(arena, newShape))).withName("reshape").withInput(tensor).withGradFunction(
                    (result, grads) -> reshape(grads, tensor.shape()));
        }
    }

    public static void release(Tensor<?, ?, ?, ?, ?> tensor) {
        handleStatus(() -> arrayfire_h.af_release_array(tensor.dereference()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> retain(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_retain_array(ptr, tensor.dereference()))
                .withName("retain")
                .withInput(tensor)
                .withGradFunction((result, grads) -> grads);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> void retainInto(
            Tensor<T, D0, D1, D2, D3> tensor, Variable<T, D0, D1, D2, D3> params) {
        release(params);
        handleStatus(() -> arrayfire_h.af_retain_array(params.segment(), tensor.dereference()));
    }

    public static int refCount(Tensor<?, ?, ?, ?, ?> tensor) {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_data_ref_count(result, tensor.dereference()));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Variable<T, D0, D1, D2, D3> variable(
            Supplier<Tensor<T, D0, D1, D2, D3>> initializer) {
        var tensor = af.tidy(initializer);
        var variable = new Variable<>(tensor.type(), tensor.shape());
        variable.segment().copyFrom(tensor.segment());
        MemoryScope.untrack(tensor);
        return variable;
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Params<T, D0, D1, D2, D3> params(
            Supplier<Tensor<T, D0, D1, D2, D3>> initializer, OptimizerProvider optimizerProvider) {
        var tensor = af.tidy(initializer);
        var params = new Params<>(tensor.type(), tensor.shape(), optimizerProvider);
        params.segment().copyFrom(tensor.segment());
        MemoryScope.untrack(tensor);
        return params;
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> eval(
            Tensor<T, D0, D1, D2, D3> tensor) {
        handleStatus(() -> arrayfire_h.af_eval(tensor.dereference()));
        return tensor;
    }

    public static void eval(Tensor<?, ?, ?, ?, ?>... tensors) {
        try (Arena arena = Arena.ofConfined()) {
            var array = arena.allocateArray(ValueLayout.ADDRESS, tensors.length);
            for (int i = 0; i < tensors.length; i++) {
                array.setAtIndex(ValueLayout.ADDRESS, i, tensors[i].dereference());
            }
            handleStatus(() -> arrayfire_h.af_eval_multiple(tensors.length, array));
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
            Tensor<T, D0, D1, D2, D3> tensor, Tileable<T, ?, ?, ?, ?> tileable) {
        checkTileableIsSmaller(tensor, tileable);
        return mul(tensor, tileable.tensor().tileAs(tensor));
    }

    private static void checkTileableIsSmaller(Tensor<?, ?, ?, ?, ?> left, Tileable<?, ?, ?, ?, ?> right) {
        if (left.d0().size() < right.tensor().d0().size() || left.d1().size() < right.tensor().d1().size() || left.d2().size() < right.tensor().d2().size() || left.d3().size() < right.tensor().d3().size()) {
            throw new IllegalArgumentException(
                    String.format("Tileable shape %s is larger than tensor shape %s", right.tensor().shape(),
                            left.shape()));
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
            Tensor<T, D0, D1, D2, D3> left, double right) {
        return mul(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return fromOperation(left.type(), left.shape(),
                ptr -> arrayfire_h.af_mul(ptr, left.dereference(), right.dereference(),
                        true)).withName("mul").withInputs(left, right).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(mul(grads, right), mul(grads, left)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> div(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return fromOperation(left.type(), left.shape(),
                ptr -> arrayfire_h.af_div(ptr, left.dereference(), right.dereference(), true)).withName(
                "div").withInputs(left, right).withGradFunction((result, resultGrads) -> {
            var rightReciprocal = af.div(af.constant(1f).cast(left.type()).tileAs(right), right);
            var leftGrads = mul(rightReciprocal, resultGrads);
            var rightGrads = af.mul(af.mul(leftGrads, left.negate()), rightReciprocal);
            return new GradFunction.TensorPair<>(leftGrads, rightGrads);
        });
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> add(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return fromOperation(left.type(), left.shape(),
                ptr -> arrayfire_h.af_add(ptr, left.dereference(), right.dereference(),
                        false)).withName("add").withInputs(left, right).withGradFunction(
                (result, resultGrads) -> new GradFunction.TensorPair<>(resultGrads, resultGrads));
    }


    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sub(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return fromOperation(left.type(), left.shape(),
                ptr -> arrayfire_h.af_sub(ptr, left.dereference(), right.dereference(),
                        true)).withName("sub").withInputs(left, right).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(grads, grads.negate()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> ge(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return fromOperation(B8, tensor.shape(),
                ptr -> arrayfire_h.af_ge(ptr, tensor.dereference(), rhs.dereference(), true)).withName(
                "ge").withInput(tensor).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> le(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return fromOperation(B8, tensor.shape(),
                ptr -> arrayfire_h.af_le(ptr, tensor.dereference(), rhs.dereference(), true)).withName(
                "le").withInput(tensor).withoutGradFunction();
    }

    public static <D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> and(
            Tensor<B8, D0, D1, D2, D3> left, Tensor<B8, D0, D1, D2, D3> right) {
        return fromOperation(B8, left.shape(),
                ptr -> arrayfire_h.af_and(ptr, left.dereference(), right.dereference(), true)).withName(
                "and").withInputs(left, right).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> maxof(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_maxof(ptr, tensor.dereference(), rhs.dereference(), true)).withName(
                "maxof").withInput(tensor).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> minof(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_minof(ptr, tensor.dereference(), rhs.dereference(), true)).withName(
                "minof").withInput(tensor).withoutGradFunction();
    }


    public static <T extends DataType<?, ?>, LD0 extends Num<LD0>, RD0 extends Num<RD0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, N, D1, D2, D3> join(
            Tensor<T, LD0, D1, D2, D3> lhs, Tensor<T, RD0, D1, D2, D3> rhs) {
        return fromOperation(lhs.type(),
                shape(n(lhs.d0().size() + rhs.d0().size()), lhs.d1(), lhs.d2(),
                        lhs.d3()),
                ptr -> arrayfire_h.af_join(ptr, 0, lhs.dereference(), rhs.dereference())).withName(
                "join(D0)").withInputs(lhs, rhs).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(index(grads, seq(lhs.d0())),
                        index(grads, seq(lhs.d0().size(), rhs.d0()))));
    }


    public static <T extends DataType<?, ?>, LD1 extends Num<LD1>, RD1 extends Num<RD1>, D0 extends Num<D0>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, N, D2, D3> join(
            Tensor<T, D0, LD1, D2, D3> lhs, Tensor<T, D0, RD1, D2, D3> rhs, arrayfire.D1 ignored) {
        assert lhs.d0().size() == rhs.d0().size() && lhs.d2().size() == rhs.d2().size() && lhs.d3().size() == rhs.d3().size() : String.format(
                "Incompatible shapes to join along d1: %s, %s", lhs.shape(), rhs.shape());
        return fromOperation(lhs.type(),
                shape(lhs.d0(), n(lhs.d1().size() + rhs.d1().size()), lhs.d2(),
                        lhs.d3()),
                ptr -> arrayfire_h.af_join(ptr, 1, lhs.dereference(), rhs.dereference())).withName(
                "join(D1)").withInputs(lhs, rhs).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(index(grads, span(), seq(lhs.d1())),
                        index(grads, span(), seq(lhs.d1().size(), rhs.d1()))));
    }

    public static <T extends DataType<?, ?>, LD2 extends Num<LD2>, RD2 extends Num<RD2>, D0 extends Num<D0>, D1 extends Num<D1>, D3 extends Num<D3>> Tensor<T, D0, D1, N, D3> join(
            Tensor<T, D0, D1, LD2, D3> lhs, Tensor<T, D0, D1, RD2, D3> rhs, arrayfire.D2 ignored) {
        assert lhs.d0().size() == rhs.d0().size() && lhs.d1().size() == rhs.d1().size() && lhs.d3().size() == rhs.d3().size() : String.format(
                "Incompatible shapes to join along d2: %s, %s", lhs.shape(), rhs.shape());
        return fromOperation(lhs.type(),
                shape(lhs.d0(), lhs.d1(), n(lhs.d2().size() + rhs.d2().size()),
                        lhs.d3()),
                ptr -> arrayfire_h.af_join(ptr, 2, lhs.dereference(), rhs.dereference())).withName(
                "join(D2)").withInputs(lhs, rhs).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(index(grads, span(), span(), seq(lhs.d2())),
                        index(grads, span(), span(), seq(lhs.d2().size(), rhs.d2()))));
    }

    public static <T extends DataType<?, ?>, LD3 extends Num<LD3>, RD3 extends Num<RD3>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>> Tensor<T, D0, D1, D2, N> join(
            Tensor<T, D0, D1, D2, LD3> lhs, Tensor<T, D0, D1, D2, RD3> rhs, arrayfire.D3 ignored) {
        assert lhs.d0().size() == rhs.d0().size() && lhs.d1().size() == rhs.d1().size() && lhs.d2().size() == rhs.d2().size() : String.format(
                "Incompatible shapes to join along d3: %s, %s", lhs.shape(), rhs.shape());
        return fromOperation(lhs.type(), shape(lhs.d0(), lhs.d1(), lhs.d2(),
                        n(lhs.d3().size() + rhs.d3().size())),
                ptr -> arrayfire_h.af_join(ptr, 3, lhs.dereference(), rhs.dereference())).withName(
                "join(D3)").withInputs(lhs, rhs).withGradFunction(
                (result, grads) -> new GradFunction.TensorPair<>(index(grads, span(), span(), span(), seq(lhs.d3())),
                        index(grads, span(), span(), span(), seq(lhs.d3().size(), rhs.d3()))));
    }


    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, U, D1, D2, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return sum(tensor, D0);
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, U, D1, D2, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType()).withGradFunction(
                (result, grads) -> grads.cast(tensor.type()).tileAs(tensor));

    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, U, D2, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType()).withGradFunction(
                (result, grads) -> grads.cast(tensor.type()).tileAs(tensor));
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, D1, U, D3> sum(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType()).withGradFunction(
                (result, grads) -> grads.cast(tensor.type()).tileAs(tensor));
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, D1, D2, U> sum(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType()).withGradFunction(
                (result, grads) -> grads.cast(tensor.type()).tileAs(tensor));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return mean(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type()).withGradFunction(
                (result, grads) -> af.div(grads.tileAs(tensor),
                        af.constant(tensor.type(), tensor.d0().size()).tileAs(tensor)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type()).withGradFunction(
                (result, grads) -> af.div(grads.tileAs(tensor),
                        af.constant(tensor.type(), tensor.d1().size()).tileAs(tensor)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type()).withGradFunction(
                (result, grads) -> af.div(grads.tileAs(tensor),
                        af.constant(tensor.type(), tensor.d2().size()).tileAs(tensor)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> mean(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type()).withGradFunction(
                (result, grads) -> af.div(grads.tileAs(tensor),
                        af.constant(tensor.type(), tensor.d3().size()).tileAs(tensor)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> median(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return median(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> median(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> median(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> median(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> median(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return max(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> max(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> max(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return min(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> min(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type()).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> min(
            Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type()).withoutGradFunction();
    }


    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> ImaxResult<T, U, D1, D2, D3> imax(
            Tensor<T, D0, D1, D2, D3> tensor) {
        try (Arena arena = Arena.ofConfined()) {
            var shape = shape(u(), tensor.d1(), tensor.d2(), tensor.d3());
            var maxValues = arena.allocate(Tensor.LAYOUT);
            var maxIndices = arena.allocate(Tensor.LAYOUT);
            handleStatus(() -> arrayfire_h.af_imax(maxValues, maxIndices, tensor.dereference(), 0));
            return new ImaxResult<>(fromSegment(tensor.type(), shape, maxValues).withName("imax_values").withInput(
                    tensor).withoutGradFunction(),
                    fromSegment(U32, shape, maxIndices).withName("imax_indices").withInput(
                            tensor).withoutGradFunction());
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, K extends Num<?>> TopKResult<T, K, D1, D2, D3> topk(
            Tensor<T, D0, D1, D2, D3> tensor, K k) {
        try (Arena arena = Arena.ofConfined()) {
            var shape = shape(k, tensor.d1(), tensor.d2(), tensor.d3());
            var topValues = arena.allocate(Tensor.LAYOUT);
            var topIndices = arena.allocate(Tensor.LAYOUT);
            handleStatus(() -> arrayfire_h.af_topk(topValues, topIndices, tensor.dereference(), k.size(), 0, 0));
            return new TopKResult<>(fromSegment(tensor.type(), shape, topValues).withName("topk_values").withInput(
                    tensor).withoutGradFunction(),
                    fromSegment(U32, shape, topIndices).withName("topk_indices").withInput(
                            tensor).withoutGradFunction());
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D0, D2, D3> diag(
            Tensor<T, D0, U, D2, D3> tensor) {
        return fromOperation(tensor.type(),
                shape(tensor.d0(), tensor.d0(), tensor.d2(), tensor.d3()),
                ptr -> arrayfire_h.af_diag_create(ptr, tensor.dereference(), 0)).withName("diag").withInput(tensor)
                // TODO: Implement grad function.
                .withoutGradFunction();
    }

    // https://arrayfire.org/docs/group__blas__func__matmul.htm
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD1 extends Num<?>> Tensor<T, D0, OD1, D2, D3> matmul(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D1, OD1, D2, D3> right) {
        assert left.d1().size() == right.d0().size() : String.format(
                "Misaligned shapes for matmul, left: %s right: %s", left.shape(), right.shape());
        return fromOperation(left.type(),
                shape(left.d0(), right.d1(), left.d2(), left.d3()),
                ptr -> arrayfire_h.af_matmul(ptr, left.dereference(), right.dereference(), 0, 0)).withName(
                "matmul").withInputs(left, right).withGradFunction((result, resultGrads) -> {
            var leftGrads = matmul(resultGrads, right.transpose());
            var rightGrads = matmul(resultGrads.transpose(), left).transpose();
            return new GradFunction.TensorPair<>(leftGrads, rightGrads);
        });
    }

    public static <T extends DataType<?, ?>, AD0 extends Num<?>, AD1 extends Num<?>, BD1 extends Num<?>, CD1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, AD0, CD1, D2, D3> matmul(
            Tensor<T, AD0, AD1, D2, D3> a, Tensor<T, AD1, BD1, D2, D3> b, Tensor<T, BD1, CD1, D2, D3> c) {
        // Determine the optimal order of operations.
        if (a.d0().size() * b.d1().size() < b.d0().size() * c.d1().size()) {
            var tmp = matmul(a, b);
            var result = matmul(tmp, c);
            tmp.dispose();
            return result;
        } else {
            var tmp = matmul(b, c);
            var result = matmul(a, tmp);
            tmp.dispose();
            return result;
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> clamp(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> lo, Tensor<T, D0, D1, D2, D3> hi) {
        // TODO: Batch parameter.
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_clamp(ptr, tensor.dereference(), lo.dereference(), hi.dereference(),
                        true)).withName("clamp").withInput(tensor).withGradFunction((result, grads) -> {
            var loMask = ge(tensor, lo);
            var hiMask = le(tensor, hi);
            return mul(grads, and(loMask, hiMask).cast(grads.type()));
        });

    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> relu(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return clamp(tensor, constant(tensor.type(), 0f).tileAs(tensor),
                constant(tensor.type(), Double.POSITIVE_INFINITY).tileAs(tensor));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> eq(
            Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return fromOperation(B8, left.shape(),
                ptr -> arrayfire_h.af_eq(ptr, left.dereference(), right.dereference(), true)).withName("eq").withInputs(
                left, right).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> negate(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var minusOne = constant(tensor.type(), tensor.shape(), -1);
        return mul(tensor, minusOne);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> exp(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_exp(ptr, tensor.dereference())).withName("exp").withInput(
                tensor).withGradFunction((result, grads) -> mul(grads, result));

    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> pow(
            Tensor<T, D0, D1, D2, D3> tensor, double pow) {
        if (pow == 2) {
            // Save some flops.
            return mul(tensor, tensor);
        }
        return pow(tensor, constant(tensor.type(), tensor.shape(), pow));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> pow(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> pow) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_pow(ptr, tensor.dereference(), pow.dereference(), false)).withName(
                "pow").withInput(tensor).withGradFunction(
                (result, grads) -> mul(mul(grads, pow), pow(tensor, sub(pow, constant(pow.type(), pow.shape(), 1)))));
    }

    /**
     * Returns 1 for negative numbers and 0 for positive numbers.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> signbit(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sign(ptr, tensor.dereference())).withName("sign").withInput(
                tensor).withoutGradFunction();
    }

    /**
     * Returns -1 for negative numbers and 1 for positive numbers.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> signum(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromTidy(() -> sub(af.constant(tensor.type(), tensor.shape(), 1),
                mul(af.constant(tensor.type(), tensor.shape(), 2), signbit(tensor)))).withName("signum").withInput(
                tensor).withoutGradFunction();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> log(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_log(ptr, tensor.dereference())).withName("log").withInput(
                tensor).withGradFunction((result, grads) -> div(grads, tensor));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> abs(
            Tensor<T, D0, D1, D2, D3> input) {
        return fromOperation(input.type(), input.shape(), ptr -> arrayfire_h.af_abs(ptr, input.dereference())).withName(
                "abs").withInput(input).withGradFunction((result, grads) -> mul(grads, signum(input)));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sqrt(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_sqrt(ptr, tensor.dereference())).withName("sqrt").withInput(
                tensor).withGradFunction(
                (result, grads) -> div(grads, mul(constant(tensor.type(), tensor.shape(), 2), result)));
    }

    public static <T extends DataType<?, T>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> softmax(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return softmax(tensor, 1f);
    }

    public static <T extends DataType<?, T>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> softmax(
            Tensor<T, D0, D1, D2, D3> tensor, float temperature) {
        return fromTidy(() -> {
            var max = max(tensor);
            var normalized = sub(tensor, max.tileAs(tensor));
            var exp = exp(div(normalized, constant(tensor.type(), tensor.shape(), temperature)));
            return div(exp, sum(exp).tileAs(tensor));
        }).withName("softmax").withInput(tensor).withGradFunction((result, grads) -> {
            // Compact all dimensions except the first into a batch dimension, so we have a spare dimension for the jacobian.
            var shape = result.shape();
            var workingShape = af.shape(shape.d0(), af.u(),
                    af.b(result.d1().size() * result.d2().size() * result.d3().size()));
            var resultTensor = result.reshape(workingShape);
            var gradsTensor = grads.reshape(workingShape);
            var positives = af.mul(resultTensor, gradsTensor);
            var negatives = af.matmul(resultTensor, resultTensor.transpose(), gradsTensor);
            var inputGrads = af.sub(positives, negatives);
            return inputGrads.reshape(tensor.shape());
        });
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sigmoid(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var one = ones(tensor);
        return div(one, add(one, exp(negate(tensor))));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sparse(
            Tensor<T, D0, D1, D2, D3> tensor, Storage storage) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_create_sparse_array_from_dense(ptr, tensor.dereference(),
                        storage.code())).withName("sparse").withInput(tensor).withGradFunction(
                (result, grads) -> grads);
    }


    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, ?, D1, D2, D3> tensor, Index<D0> i0) {
        return index(tensor, i0, seq(tensor.d1()), seq(tensor.d2()), seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, ?, ?, D2, D3> tensor, Index<D0> i0, Index<D1> i1) {
        return index(tensor, i0, i1, seq(tensor.d2()), seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, D0, ?, D2, D3> tensor, Span ignored0, Index<D1> i1) {
        return index(tensor, seq(tensor.d0()), i1, seq(tensor.d2()), seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, D0, D1, ?, D3> tensor, Span ignored0, Span ignored1, Index<D2> i2) {
        return index(tensor, seq(tensor.d0()), seq(tensor.d1()), i2, seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, D0, ?, ?, D3> tensor, Span ignored0, Index<D1> i1, Index<D2> i2) {
        return index(tensor, seq(tensor.d0()), i1, i2, seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, ?, D1, ?, D3> tensor, Index<D0> i0, Span ignored1, Index<D2> i2) {
        return index(tensor, i0, seq(tensor.d1()), i2, seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, ?, ?, ?, D3> tensor, Index<D0> i0, Index<D1> i1, Index<D2> i2) {
        return index(tensor, i0, i1, i2, seq(tensor.d3()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, D0, D1, D2, ?> tensor, Span ignored0, Span ignored1, Span ignored2, Index<D3> i3) {
        return index(tensor, seq(tensor.d0()), seq(tensor.d1()), seq(tensor.d2()), i3);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, D1, D2, D3> index(
            Tensor<T, ?, ?, ?, ?> tensor, Index<D0> i0, Index<D1> i1, Index<D2> i2, Index<D3> i3) {
        try (Arena arena = Arena.ofConfined()) {
            var layout = MemoryLayout.sequenceLayout(4, Index.LAYOUT);
            var nativeIndexes = arena.allocateArray(Index.LAYOUT, 4);

            i0.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(0)),
                    Index.LAYOUT.byteSize()));
            i1.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(1)),
                    Index.LAYOUT.byteSize()));
            i2.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(2)),
                    Index.LAYOUT.byteSize()));
            i3.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(3)),
                    Index.LAYOUT.byteSize()));

            var result = arena.allocate(ValueLayout.ADDRESS);
            handleStatus(() -> arrayfire_h.af_index_gen(result, tensor.dereference(), 4, nativeIndexes));

            // We don't obviously know the new shape, so we need to compute it.
            var dims = arena.allocateArray(ValueLayout.JAVA_LONG, 4);
            handleStatus(
                    () -> arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                            result.get(ValueLayout.ADDRESS, 0)));
            var d0 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 0);
            var d1 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 1);
            var d2 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 2);
            var d3 = (int) dims.getAtIndex(ValueLayout.JAVA_LONG, 3);

            return fromSegment(tensor.type(),
                    shape(i0.createDim(d0), i1.createDim(d1), i2.createDim(d2), i3.createDim(d3)), result).withName(
                    "index").withInput(tensor).withoutGradFunction();

        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> List<Tensor<T, D0, N, U, U>> batch(
            Tensor<T, D0, D1, U, U> tensor, int batchSize) {
        return batch(tensor, ArrayFire::n, batchSize);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, BDT extends Num<BDT>> List<Tensor<T, D0, BDT, U, U>> batch(
            Tensor<T, D0, D1, D2, D3> tensor, Function<Integer, BDT> type, int batchSize) {
        var results = new ArrayList<Tensor<T, D0, BDT, U, U>>();
        var d0Seq = seq(tensor.d0());
        for (int i = 0; i < tensor.d1().size(); i += batchSize) {
            var computedD1Size = Math.min(batchSize, tensor.d1().size() - i);
            var slice = index(tensor, d0Seq, seq(i, i + computedD1Size - 1));
            results.add(slice.reshape(shape(tensor.d0(), type.apply(computedD1Size))));
        }
        return results;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<T, OD0, OD1, OD2, OD3> tileAs(
            Tensor<T, D0, D1, D2, D3> tensor, Shape<OD0, OD1, OD2, OD3> newShape) {
        assert newShape.capacity() % tensor.shape().capacity() == 0 : String.format(
                "Can't tile perfectly from %s to %s", tensor.shape(), newShape);
        int d0ratio = newShape.d0().size() / tensor.d0().size();
        int d1ratio = newShape.d1().size() / tensor.d1().size();
        int d2ratio = newShape.d2().size() / tensor.d2().size();
        int d3ratio = newShape.d3().size() / tensor.d3().size();
        return fromOperation(tensor.type(), newShape,
                ptr -> arrayfire_h.af_tile(ptr, tensor.dereference(), d0ratio, d1ratio, d2ratio, d3ratio)).withName(
                "tile").withInput(tensor).withGradFunction(
                (result, grads) -> sumAs((Tensor) grads, tensor.shape()).cast(tensor.type()));
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<ST, OD0, OD1, OD2, OD3> sumAs(
            Tensor<T, D0, D1, D2, D3> input, Shape<OD0, OD1, OD2, OD3> newShape) {
        // I think there is a nicer way to do this in at most two operations.
        Tensor result = input;
        if (newShape.d0() != input.d0()) {
            if (newShape.d0().size() != 1)
                throw new IllegalArgumentException("Can't sum over D0 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d1() != input.d1()) {
            if (newShape.d1().size() != 1)
                throw new IllegalArgumentException("Can't sum over D1 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d2() != input.d2()) {
            if (newShape.d2().size() != 1)
                throw new IllegalArgumentException("Can't sum over D2 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d3() != input.d3()) {
            if (newShape.d3().size() != 1)
                throw new IllegalArgumentException("Can't sum over D3 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        return ((Tensor<ST, ?, ?, ?, ?>) result).reshape(newShape);
    }

    public static <T extends DataType<?, ?>> Tensor<T, N, U, U, U> flatten(Tensor<T, ?, ?, ?, ?> tensor) {
        return reshape(tensor, shape(tensor.shape().capacity()));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> flip(
            Tensor<T, D0, D1, D2, D3> tensor) {
        // TODO: Investigate fixed parameters.
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_flip(ptr, tensor.dereference(), 0)).withName("flip").withInput(
                tensor).withGradFunction((result, grads) -> flip(grads));
    }


    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, FD3 extends Num<?>> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters) {
        return convolve2(tensor, filters, shape(1, 1), shape(0, 0), shape(1, 1));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, FD3 extends Num<?>> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride) {
        return convolve2(tensor, filters, stride, shape(0, 0), shape(1, 1));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, FD3 extends Num<?>> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride,
            Shape<?, ?, ?, ?> padding) {
        return convolve2(tensor, filters, stride, padding, shape(1, 1));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, FD3 extends Num<?>> Tensor<T, N, N, FD3, D3> convolve2(
            Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, ?, ?, D2, FD3> filters, Shape<?, ?, ?, ?> stride,
            Shape<?, ?, ?, ?> padding, Shape<?, ?, ?, ?> dilation) {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.ADDRESS);
            // TODO(https://github.com/arrayfire/arrayfire/issues/3402): Convolutions look like they may allocate memory outside of ArrayFire's scope, so sometimes we need to GC first.
            retryWithGc(() -> handleStatus(
                    () -> arrayfire_h.af_convolve2_nn(result, tensor.dereference(), filters.dereference(), 2,
                            nativeDims(arena, stride), 2, nativeDims(arena, padding), 2, nativeDims(arena, dilation))));
            var computedDims = getDims(result);
            return fromSegment(tensor.type(),
                    shape(n((int) computedDims[0]), n((int) computedDims[1]), filters.d3(),
                            tensor.d3()), result).withName("convolve2").withInputs(tensor,
                    filters).withoutGradFunction();
        }
    }

    /**
     * L2 norm.
     */
    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, U, D1, D2, D3> norm(
            Tensor<T, D0, D1, D2, D3> tensor) {
        var mul = mul(tensor, tensor);
        var sum = sum(mul);
        return sqrt(sum);
    }

    /**
     * Normalize by dividing by the L2 norm.
     */
    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, D1, D2, D3> normalize(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return div(cast(tensor, tensor.type().sumType()), norm(tensor).tileAs(tensor.shape()));
    }

    /**
     * Center by subtracting the average.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> center(
            Tensor<T, D0, D1, D2, D3> tensor) {
        return sub(tensor, mean(tensor).tileAs(tensor));
    }

    // svd
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>> SvdResult<T, D0, D1> svd(
            Tensor<T, D0, D1, U, U> tensor) {
        try (Arena arena = Arena.ofConfined()) {
            var u = arena.allocate(Tensor.LAYOUT);
            var s = arena.allocate(Tensor.LAYOUT);
            var v = arena.allocate(Tensor.LAYOUT);
            handleStatus(() -> arrayfire_h.af_svd(u, s, v, tensor.dereference()));
            return new SvdResult<>(
                    fromSegment(tensor.type(), shape(tensor.d0(), tensor.d0()), u).withName(
                            "svd_u").withInput(tensor).withoutGradFunction(),
                    fromSegment(tensor.type(), shape(tensor.d0()), s).withName("svd_s").withInput(
                            tensor).withoutGradFunction(),
                    fromSegment(tensor.type(), shape(tensor.d1(), tensor.d1()), v).withName(
                            "svd_v").withInput(tensor).withoutGradFunction());
        }
    }

    /**
     * Computes the covariance matrix of the given matrix.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>> Tensor<T, D0, D0, U, U> cov(
            Tensor<T, D0, D1, U, U> tensor) {
        return tidy(() -> {
            var subMean = sub(tensor, mean(tensor, D1).tileAs(tensor));
            var matrix = matmul(subMean, subMean.transpose());
            return div(matrix, constant(matrix.type(), matrix.shape(), tensor.d1().size() - 1.0f));
        });
    }

    /**
     * Computes the ZCA whitening matrix of the given matrix.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>> Tensor<T, D0, D0, U, U> zca(
            Tensor<T, D0, D1, U, U> tensor) {
        return tidy(() -> {
            var cov = cov(tensor);
            var svd = svd(cov);
            var invSqrtS = diag(div(constant(svd.s().type(), svd.s().shape(), 1.0f),
                    sqrt(add(svd.s(), constant(svd.s().type(), svd.s().shape(), 1e-5f)))));
            return matmul(svd.u(), matmul(invSqrtS, svd.u().transpose()));
        });
    }


    /**
     * Inverts the given matrix.
     */
    public static <T extends DataType<?, ?>, D extends Num<?>> Tensor<T, D, D, U, U> inverse(
            Tensor<T, D, D, U, U> tensor) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_inverse(ptr, tensor.dereference(), 0)).withName("inverse").withInput(
                tensor).withoutGradFunction();
    }

    // TODO: Add uncropped version.
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>> Tensor<T, D0, D1, U, U> rotate(
            Tensor<T, D0, D1, U, U> tensor, float angle, InterpolationType interpolationType) {
        return fromOperation(tensor.type(), tensor.shape(),
                ptr -> arrayfire_h.af_rotate(ptr, tensor.dereference(), angle, true,
                        interpolationType.code())).withName("rotate").withInput(tensor).withGradFunction(
                (result, grads) -> rotate(grads, -angle, interpolationType));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, ND0 extends Num<?>, ND1 extends Num<?>> Tensor<T, ND0, ND1, U, U> scale(
            Tensor<T, D0, D1, U, U> tensor, ND0 nd0, ND1 nd1, InterpolationType interpolationType) {
        return fromOperation(tensor.type(), shape(nd0, nd1, tensor.d2(), tensor.d3()),
                ptr -> arrayfire_h.af_scale(ptr, tensor.dereference(), (float) nd0.size() / tensor.d0().size(),
                        (float) nd1.size() / tensor.d1().size(), nd0.size(), nd1.size(),
                        interpolationType.code())).withName("scale").withInput(tensor).withGradFunction(
                (result, grads) -> scale(grads, tensor.d0(), tensor.d1(), interpolationType));
    }

    public static void printMeminfo() {
        try (Arena arena = Arena.ofConfined()) {
            var chars = arena.allocateArray(ValueLayout.JAVA_BYTE, 1);
            handleStatus(() -> arrayfire_h.af_print_mem_info(chars, -1));
        }
    }

    public static void deviceGc() {
        handleStatus(arrayfire_h::af_device_gc);
    }

    public static A a(int value) {
        return new A(value);
    }

    public static B b(int value) {
        return new B(value);
    }

    public static C c(int value) {
        return new C(value);
    }

    public static D d(int value) {
        return new D(value);
    }

    public static E e(int value) {
        return new E(value);
    }

    public static F f(int value) {
        return new F(value);
    }

    public static G g(int value) {
        return new G(value);
    }

    public static H h(int value) {
        return new H(value);
    }

    public static I i(int value) {
        return new I(value);
    }

    public static J j(int value) {
        return new J(value);
    }

    public static K k(int value) {
        return new K(value);
    }

    public static L l(int value) {
        return new L(value);
    }

    public static M m(int value) {
        return new M(value);
    }

    public static N n(int value) {
        return new N(value);
    }

    public static O o(int value) {
        return new O(value);
    }

    public static P p(int value) {
        return new P(value);
    }

    public static Q q(int value) {
        return new Q(value);
    }

    public static R r(int value) {
        return new R(value);
    }

    public static S s(int value) {
        return new S(value);
    }

    public static T t(int value) {
        return new T(value);
    }

    public static V v(int value) {
        return new V(value);
    }

    public static W w(int value) {
        return new W(value);
    }

    public static X x(int value) {
        return new X(value);
    }

    public static Y y(int value) {
        return new Y(value);
    }

    public static Z z(int value) {
        return new Z(value);
    }

    public static final U U = new U(1);

    public static U u() {
        return U;
    }

    public static U u(int value) {
        return U;
    }

    public static void optimize(Tensor<?, ?, ?, ?, ?> loss) {
        scope().graph().optimize(loss);
    }

    private static void retryWithGc(Runnable fn) {
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

    private static void handleStatus(Supplier<Object> res) {
        maybeLoadNativeLibraries();
        var result = Status.fromCode((int) res.get());
        if (!Status.AF_SUCCESS.equals(result)) {
            throw new ArrayFireException(result);
            //      String lastError;
            //      try {
            //        lastError = lastError();
            //
            //      } catch (Exception e) {
            //        throw new RuntimeException("ArrayFireError: " + result.name());
            //      }
            //      throw new RuntimeException("ArrayFireError: " + result.name() + ": " + lastError);
        }
    }

    public DeviceMemInfo deviceMemInfo() {
        try (Arena arena = Arena.ofConfined()) {
            var allocBytes = arena.allocateArray(ValueLayout.JAVA_LONG, 1);
            var allocBuffers = arena.allocateArray(ValueLayout.JAVA_LONG, 1);
            var lockBytes = arena.allocateArray(ValueLayout.JAVA_LONG, 1);
            var lockBuffers = arena.allocateArray(ValueLayout.JAVA_LONG, 1);
            handleStatus(() -> arrayfire_h.af_device_mem_info(allocBytes, allocBuffers, lockBytes, lockBuffers));
            return new DeviceMemInfo(allocBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
                    allocBuffers.getAtIndex(ValueLayout.JAVA_LONG, 0), lockBytes.getAtIndex(ValueLayout.JAVA_LONG, 0),
                    lockBuffers.getAtIndex(ValueLayout.JAVA_LONG, 0));
        }
    }

    public static MemorySegment allocPinned(long bytes) {
        try (Arena arena = Arena.ofConfined()) {
            var ptr = arena.allocateArray(ValueLayout.ADDRESS, 1);
            handleStatus(() -> arrayfire_h.af_alloc_pinned(ptr, bytes));
            return MemorySegment.ofAddress(ptr.getAtIndex(ValueLayout.ADDRESS, 0).address()).reinterpret(bytes);
        }
    }

    public static void freePinned(MemorySegment segment) {
        handleStatus(() -> arrayfire_h.af_free_pinned(segment));
    }
}
