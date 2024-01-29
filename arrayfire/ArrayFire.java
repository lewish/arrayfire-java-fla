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
    public static final U U = new U(1);
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
                Scope.move(mc, parentScope);
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

    /**
     * Sorts a tensor over D0.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
        Tensor<DT, D0, D1, D2, D3> tensor) {
        return sort(tensor, D0);
    }

    /**
     * Sorts a tensor over the given dimension.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
        Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sort(tensor, dim, true);
    }

    /**
     * Sorts a tensor over the given dimension in ascending or descending order.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> sort(
        Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        return operation("sort")
                   .inputs(tensor)
                   .outputs(tensor.prototype())
                   .operation(ptr -> arrayfire_h.af_sort(ptr, tensor.dereference(), dim.index(), ascending))
                   .build();
    }

    /**
     * Returns a prototype tensor with the given type and shape.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Prototype<T, D0, D1, D2, D3> prototype(
        T type, Shape<D0, D1, D2, D3> shape) {
        return new Prototype<>(type, shape);
    }

    /**
     * Returns a prototype tensor with the same type and shape as the given tensor.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Prototype<T, D0, D1, D2, D3> prototype(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return new Prototype<>(tensor.type(), tensor.shape());
    }

    /**
     * Sorts a tensor over D0 and returns the values and indices of original values.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
        Tensor<DT, D0, D1, D2, D3> tensor) {
        return sortIndex(tensor, D0);
    }

    /**
     * Sorts a tensor over the given dimension and returns the values and indices of original values.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
        Tensor<DT, D0, D1, D2, D3> tensor, Dim dim) {
        return sortIndex(tensor, dim, true);
    }

    /**
     * Sorts a tensor over the given dimension in ascending or descending order and returns the values and indices of original values.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> SortIndexResult<DT, D0, D1, D2, D3> sortIndex(
        Tensor<DT, D0, D1, D2, D3> tensor, Dim dim, boolean ascending) {
        var pair = operation("sort_index")
                       .inputs(tensor)
                       .outputs(prototype(tensor.type(), tensor.shape()), prototype(U32, tensor.shape()))
                       .operation(
                           (leftPtr, rightPtr) -> arrayfire_h.af_sort_index(leftPtr, rightPtr, tensor.dereference(),
                               dim.index(), ascending))
                       .build();
        return new SortIndexResult<>(pair.left(), pair.right());
    }

    /**
     * Create a random permutation of indices for the given dimension.
     */
    public static <D extends Num<D>> Index<D> permutation(D dim) {
        var indices = tidy(() -> sortIndex(randu(U32, shape(dim))).indices());
        return af.seq(indices);
    }

    /**
     * Creates a device tensor from the given native array.
     */
    public static <DT extends DataType<?, ?>, AT extends NativeArray<DT, ?, ?>> Tensor<DT, N, U, U, U> create(
        AT array) {
        return create(array, shape(n(array.length())));
    }

    /**
     * Creates a device tensor from the given native array and shape.
     */
    public static <DT extends DataType<?, ?>, AT extends NativeArray<DT, ?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> create(
        AT array, Shape<D0, D1, D2, D3> shape) {
        return operation("create")
                   .inputs()
                   .outputs(prototype(array.type(), shape))
                   .operation(
                       ptr -> arrayfire_h.af_create_array(ptr, array.segment(), shape.dims().length, nativeDims(shape),
                           array.type().code()))
                   .build();
    }

    /**
     * Creates a device tensor from the given type and java values.
     * This is not recommended in a production setting, as memory will be copied twice. Instead, use {@link #create(NativeArray)}.
     */
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

    /**
     * Creates a device tensor from the given type and java native array.
     * This is not recommended in a production setting, as memory will be copied twice. Instead, use {@link #create(NativeArray)}.
     */
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

    /**
     * Creates a {@link F32} device tensor from the given float values.
     */
    public static Tensor<F32, N, U, U, U> create(float... values) {
        return create(F32, values);
    }

    /**
     * Creates a {@link F64} device tensor from the given double values.
     */
    public static Tensor<F64, N, U, U, U> create(double... values) {
        return create(F64, values);
    }

    /**
     * Creates a {@link S32} device tensor from the given byte values.
     */
    public static Tensor<S32, N, U, U, U> create(int... values) {
        return create(S32, values);
    }

    /**
     * Creates a constant scalar {@link F32} device tensor from the given float value.
     */
    public static Tensor<F32, U, U, U, U> constant(float value) {
        return constant(F32, value);
    }

    /**
     * Creates a constant scalar {@link F64} device tensor from the given float value.
     */
    public static Tensor<F64, U, U, U, U> constant(double value) {
        return constant(F64, value);
    }

    /**
     * Creates a constant scalar device tensor from the given type and double value.
     */
    public static <DT extends DataType<?, ?>> Tensor<DT, U, U, U, U> constant(DT type, double value) {
        return constant(type, shape(u()), value);
    }

    /**
     * Creates a constant device tensor from the given type, shape, and double value.
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<DT, D0, D1, D2, D3> constant(
        DT type, Shape<D0, D1, D2, D3> shape, double value) {
        return operation("constant")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(
                       ptr -> arrayfire_h.af_constant(ptr, value, shape.dims().length, nativeDims(shape), type.code()))
                   .build();

    }

    /**
     * Blocks until all operations on device are finished
     */
    public static void sync() {
        handleStatus(() -> arrayfire_h.af_sync(deviceId()));
    }

    /**
     * Returns an index from the given sequence.
     */
    public static Index<N> seq(int begin, int endInclusive, int step) {
        return new Index<>(new Seq(begin, endInclusive, step), ArrayFire::n);
    }

    /**
     * Returns an index from the given sequence with a step of 1.
     */
    public static Index<N> seq(int begin, int endInclusive) {
        return new Index<>(new Seq(begin, endInclusive, 1), ArrayFire::n);
    }

    /**
     * Returns a lookup index using the given tensor as lookup values (indices).
     */
    public static <DT extends DataType<?, ?>, D0 extends Num<D0>> Index<D0> seq(Tensor<DT, D0, U, U, U> index) {
        return new Index<>(index, index.d0()::create);
    }

    /**
     * Returns a sequence from the given dimension, starting at 0 with a length equal to the dimension and a step of 1.
     */
    public static <D extends Num<D>> Index<D> seq(D num) {
        return new Index<>(new Seq(0, num.size() - 1, 1), num::create);
    }

    /**
     * Returns a sequence from the given dimension, starting at the given offset with a length equal to the dimension and a step of 1.
     */
    public static <D extends Num<D>> Index<D> seq(int offset, D num) {
        return new Index<>(new Seq(offset, num.size() - 1 + offset, 1), num::create);
    }

    /**
     * Returns a span that can be used to index the entire dimension when calling {@link #index}.
     */
    public static Span span() {
        return new Span();
    }

    /**
     * Returns a 1D shape of the given size and type N.
     */
    public static Shape<N, U, U, U> shape(int d0) {
        return new Shape<>(n(d0), u(), u(), u());
    }

    /**
     * Returns a 1D shape of the given dimension.
     */
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

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Operation.Builder.Unary<IT, D0, D1, D2, D3>.Single<T, U, D1, D2, D3> reduce(
        String name, Tensor<IT, D0, D1, D2, D3> a,
        Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D0 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(u(), a.d1(), a.d2(), a.d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Operation.Builder.Unary<IT, D0, D1, D2, D3>.Single<T, D0, U, D2, D3> reduce(
        String name, Tensor<IT, D0, D1, D2, D3> a,
        Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D1 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.d0(), u(), a.d2(), a.d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));

    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Operation.Builder.Unary<IT, D0, D1, D2, D3>.Single<T, D0, D1, U, D3> reduce(
        String name, Tensor<IT, D0, D1, D2, D3> a,
        Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D2 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.d0(), a.d1(), u(), a.d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    private static <T extends DataType<?, ?>, IT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Operation.Builder.Unary<IT, D0, D1, D2, D3>.Single<T, D0, D1, D2, U> reduce(
        String name, Tensor<IT, D0, D1, D2, D3> a,
        Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method, arrayfire.D3 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.d0(), a.d1(), a.d2(), u())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    /**
     * Cast the given tensor to the given type.
     */
    @SuppressWarnings("unchecked")
    public static <T extends DataType<?, ?>, OT extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<OT, D0, D1, D2, D3> cast(
        Tensor<T, D0, D1, D2, D3> input, OT type) {
        if (input.type().equals(type)) {
            return (Tensor<OT, D0, D1, D2, D3>) input;
        }
        return operation("cast")
                   .inputs(input)
                   .outputs(prototype(type, input.shape()))
                   .operation(ptr -> arrayfire_h.af_cast(ptr, input.dereference(), type.code()))
                   .grads((result, grads) -> grads.cast(input.type()))
                   .build();
    }

    /**
     * Returns a tensor of value 1 with the same type and shape as the given tensor.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> ones(
        Tensor<T, D0, D1, D2, D3> model) {
        return ones(model.type(), model.shape());
    }

    /**
     * Returns a tensor of value 1 with the given type and shape.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> ones(
        T type, Shape<D0, D1, D2, D3> shape) {
        return constant(type, 1).tileAs(shape);
    }

    /**
     * Returns a tensor of value 0 with the given type and shape.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> zeros(
        T type, Shape<D0, D1, D2, D3> shape) {
        return constant(type, 0).tileAs(shape);
    }

    /**
     * Create a random tensor sampled from uniform distribution between [0, 1].
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> randu(
        T type, Shape<D0, D1, D2, D3> shape) {
        return operation("randu")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_randu(ptr, shape.dims().length, nativeDims(shape), type.code()))
                   .build();
    }

    /**
     * Create a random tensor sampled from a normal distribution with mean 0.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> randn(
        T type, Shape<D0, D1, D2, D3> shape) {
        return operation("randn")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_randn(ptr, shape.dims().length, nativeDims(shape), type.code()))
                   .build();
    }

    /**
     * Create a tensor with values [0, n-1].
     */
    public static Tensor<U32, N, U, U, U> range(int n) {
        return range(U32, n);
    }

    /**
     * Create a tensor with values [0, n-1] of the given type.
     */
    public static <T extends DataType<?, ?>> Tensor<T, N, U, U, U> range(T type, int n) {
        var shape = shape(n(n));
        return operation("range")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_range(ptr, shape.dims().length, nativeDims(shape), 0, type.code()))
                   .build();
    }

    /**
     * Set the seed of the random engine.
     */
    public static void setSeed(long seed) {
        handleStatus(() -> arrayfire_h.af_set_seed(seed));
    }

    /**
     * Set the type of the random engine.
     */
    public static void setRandomEngineType(RandomEngineType type) {
        handleStatus(() -> arrayfire_h.af_set_default_random_engine_type(type.code()));
    }

    /**
     * Pull data from the device to the host, returning a native array.
     */
    public static <AT extends NativeArray<?, ?, ?>, T extends DataType<AT, ?>> AT data(Tensor<T, ?, ?, ?, ?> a) {
        var result = a.type().create(a.capacity());
        handleStatus(() -> arrayfire_h.af_get_data_ptr(result.segment(), a.dereference()));
        return result;
    }

    private static void checkDims(Tensor<?, ?, ?, ?, ?> tensor) {
        try (Arena arena = Arena.ofConfined()) {
            var dims = arena.allocateArray(ValueLayout.JAVA_LONG, 4);
            handleStatus(
                () -> arrayfire_h.af_get_dims(dims.asSlice(0), dims.asSlice(8), dims.asSlice(16), dims.asSlice(24),
                    tensor.dereference()));
            var trueDims = dims.toArray(ValueLayout.JAVA_LONG);
            var expectedDims = tensor.shape().dims();
            for (int i = 0; i < trueDims.length; i++) {
                if (trueDims[i] != expectedDims[i]) {
                    throw new IllegalStateException(
                        String.format("Expected dimensions %s but got %s", Arrays.toString(expectedDims),
                            Arrays.toString(trueDims)));
                }
            }
        }
    }

    /**
     * Return the current version of arrayfire.
     */
    public static Version version() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocateArray(ValueLayout.JAVA_INT, 3);
            handleStatus(() -> arrayfire_h.af_get_version(result, result.asSlice(4), result.asSlice(8)));
            var arr = result.toArray(ValueLayout.JAVA_INT);
            return new Version(arr[0], arr[1], arr[2]);
        }
    }

    /**
     * Return a set of the available backends. See {@link Backend} for all options.
     */
    public static Set<Backend> availableBackends() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_available_backends(result));
            return Backend.fromBitmask(result.get(ValueLayout.JAVA_INT, 0));
        }
    }

    /**
     * Return the currently active backend.
     */
    public static Backend backend() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_active_backend(result));
            return Backend.fromCode(result.get(ValueLayout.JAVA_INT, 0));
        }
    }

    /**
     * Set the backend to use for all operations.
     */
    public static void setBackend(Backend backend) {
        handleStatus(() -> arrayfire_h.af_set_backend(backend.code()));
    }

    /**
     * Return the currently active device ID.
     */
    public static int deviceId() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_device(result));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    /**
     * Set the device ID to use for all operations, see {@link #deviceCount()} for available devices.
     */
    public static void setDeviceId(int device) {
        handleStatus(() -> arrayfire_h.af_set_device(device));
    }

    /**
     * Return the number of devices available.
     */
    public static int deviceCount() {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_device_count(result));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    /**
     * Return information about the currently active device.
     */
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

    private static MemorySegment nativeDims(Shape<?, ?, ?, ?> shape) {
        return Arena.ofAuto().allocateArray(ValueLayout.JAVA_LONG, shape.dims());
    }

    /**
     * Transpose D0 and D1 dimensions of the given tensor.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D1, D0, D2, D3> transpose(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("transpose")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), shape(tensor.d1(), tensor.d0(), tensor.d2(), tensor.d3())))
                   .operation(ptr -> arrayfire_h.af_transpose(ptr, tensor.dereference(), true))
                   .grads((result, grads) -> transpose(grads))
                   .build();
    }

    /**
     * Change the type of the tensor's D0 dimension to the given type variable provider.
     */
    public static <T extends DataType<?, ?>, OD0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, D1, D2, D3> castshape(
        Tensor<T, ?, D1, D2, D3> tensor, Function<Integer, OD0> d0) {
        return reshape(tensor, shape(d0.apply(tensor.d0().size()), tensor.d1(), tensor.d2(), tensor.d3()));
    }

    /**
     * Change the type of the tensor's D0, D1 dimensions to the given type variable providers.
     */
    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, OD1, D2, D3> castshape(
        Tensor<T, ?, ?, D2, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1) {
        return reshape(tensor,
            shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()), tensor.d2(), tensor.d3()));
    }

    /**
     * Change the type of the tensor's D0, D1, D2 dimensions to the given type variable providers.
     */
    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, D3 extends Num<?>> Tensor<T, OD0, OD1, OD2, D3> castshape(
        Tensor<T, ?, ?, ?, D3> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
        Function<Integer, OD2> d2) {
        return reshape(tensor,
            shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()), d2.apply(tensor.d2().size()),
                tensor.d3()));
    }

    /**
     * Change the type of the tensor's dimensions to the given type variable providers.
     */
    public static <T extends DataType<?, ?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<T, OD0, OD1, OD2, OD3> castshape(
        Tensor<T, ?, ?, ?, ?> tensor, Function<Integer, OD0> d0, Function<Integer, OD1> d1, Function<Integer, OD2> d2,
        Function<Integer, OD3> d3) {
        return reshape(tensor,
            shape(d0.apply(tensor.d0().size()), d1.apply(tensor.d1().size()), d2.apply(tensor.d2().size()),
                d3.apply(tensor.d3().size())));
    }

    /**
     * Reshape the tensor to the given shape.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD0 extends Num<?>, OD1 extends Num<?>, OD2 extends Num<?>, OD3 extends Num<?>> Tensor<T, OD0, OD1, OD2, OD3> reshape(
        Tensor<T, D0, D1, D2, D3> tensor, Shape<OD0, OD1, OD2, OD3> newShape) {
        if (tensor.shape().capacity() != newShape.capacity()) {
            throw new IllegalArgumentException(
                String.format("New shape %s doesn't have same capacity as original shape %s", newShape,
                    tensor.shape()));
        }
        return operation("reshape")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), newShape))
                   .operation(ptr -> arrayfire_h.af_moddims(ptr, tensor.dereference(), newShape.dims().length,
                       nativeDims(newShape)))
                   .grads((result, grads) -> reshape(grads, tensor.shape()))
                   .build();
    }

    /**
     * Release the memory of the given tensor on the device.
     */
    public static void release(Tensor<?, ?, ?, ?, ?> tensor) {
        handleStatus(() -> arrayfire_h.af_release_array(tensor.dereference()));
        Scope.untrack(tensor);
    }

    /**
     * Retain the given tensor, increasing its ref count by 1 and return a new container for it.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> retain(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("retain")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), tensor.shape()))
                   .operation(ptr -> arrayfire_h.af_retain_array(ptr, tensor.dereference()))
                   .grads((result, grads) -> grads)
                   .build();
    }

    /**
     * Set the values of the given variable to the values of the given tensor.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Operation set(
        Variable<T, D0, D1, D2, D3> variable, Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("set").inputs(tensor).outputs().operation(() -> {
            handleStatus(() -> arrayfire_h.af_release_array(variable.dereference()));
            handleStatus(() -> arrayfire_h.af_retain_array(variable.segment(), tensor.dereference()));
        }).build();
    }

    /**
     * Return the ref count of the given tensor.
     */
    public static int refCount(Tensor<?, ?, ?, ?, ?> tensor) {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_data_ref_count(result, tensor.dereference()));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    /**
     * Create a variable with the given initializer.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Variable<T, D0, D1, D2, D3> variable(
        Supplier<Tensor<T, D0, D1, D2, D3>> initializer) {
        var tensor = af.tidy(initializer);
        var variable = new Variable<>(tensor.type(), tensor.shape());
        variable.segment().copyFrom(tensor.segment());
        Scope.untrack(tensor);
        return variable;
    }

    /**
     * Create params with the given initializer and optimizer.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Params<T, D0, D1, D2, D3> params(
        Supplier<Tensor<T, D0, D1, D2, D3>> initializer, OptimizerProvider optimizerProvider) {
        var tensor = af.tidy(initializer);
        var params = new Params<>(tensor.type(), tensor.shape(), optimizerProvider);
        params.segment().copyFrom(tensor.segment());
        Scope.untrack(tensor);
        return params;
    }

    /**
     * Evaluate the tensor, telling the ArrayFire JIT compiler that you want the literal values of the tensor.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> eval(
        Tensor<T, D0, D1, D2, D3> tensor) {
        handleStatus(() -> arrayfire_h.af_eval(tensor.dereference()));
        return tensor;
    }

    /**
     * Evaluate the tensors, telling the ArrayFire JIT compiler that you want the literal values of the tensors.
     */
    public static void eval(Tensor<?, ?, ?, ?, ?>... tensors) {
        try (Arena arena = Arena.ofConfined()) {
            var array = arena.allocateArray(ValueLayout.ADDRESS, tensors.length);
            for (int i = 0; i < tensors.length; i++) {
                array.setAtIndex(ValueLayout.ADDRESS, i, tensors[i].dereference());
            }
            handleStatus(() -> arrayfire_h.af_eval_multiple(tensors.length, array));
        }
    }

    /**
     * Multiply two tensors together element wise, broadcasting the smaller tensor to the larger tensor's shape.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
        Tensor<T, D0, D1, D2, D3> tensor, Tileable<T, ?, ?, ?, ?> tileable) {
        checkTileableIsSmaller(tensor, tileable);
        return mul(tensor, tileable.tensor().tileAs(tensor));
    }

    private static void checkTileableIsSmaller(Tensor<?, ?, ?, ?, ?> left, Tileable<?, ?, ?, ?, ?> right) {
        if (left.d0().size() < right.tensor().d0().size() || left.d1().size() < right.tensor().d1().size() ||
                left.d2().size() < right.tensor().d2().size() || left.d3().size() < right.tensor().d3().size()) {
            throw new IllegalArgumentException(
                String.format("Tileable shape %s is larger than tensor shape %s", right.tensor().shape(),
                    left.shape()));
        }
    }

    /**
     * Multiply the tensor by a scalar value.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
        Tensor<T, D0, D1, D2, D3> left, double right) {
        return mul(left, af.constant(left.type(), left.shape(), right));
    }

    /**
     * Multiply two tensors together, element wise.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> mul(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("mul")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_mul(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(mul(grads, right), mul(grads, left)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> div(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("div")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_div(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> {
                       var rightReciprocal = af.div(af.constant(1f).cast(left.type()).tileAs(right), right);
                       var leftGrads = mul(rightReciprocal, grads);
                       var rightGrads = af.mul(af.mul(leftGrads, left.negate()), rightReciprocal);
                       return new GradFunction.TensorPair<>(leftGrads, rightGrads);
                   })
                   .build();

    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> add(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("add")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_add(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(grads, grads))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sub(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("sub")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_sub(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(grads, grads.negate()))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> ge(
        Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return operation("ge")
                   .inputs(tensor, rhs)
                   .outputs(prototype(B8, tensor.shape()))
                   .operation(ptr -> arrayfire_h.af_ge(ptr, tensor.dereference(), rhs.dereference(), true))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> le(
        Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> rhs) {
        return operation("le")
                   .inputs(tensor, rhs)
                   .outputs(prototype(B8, tensor.shape()))
                   .operation(ptr -> arrayfire_h.af_le(ptr, tensor.dereference(), rhs.dereference(), true))
                   .build();
    }

    public static <D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> and(
        Tensor<B8, D0, D1, D2, D3> left, Tensor<B8, D0, D1, D2, D3> right) {
        return operation("and")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_and(ptr, left.dereference(), right.dereference(), true))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> maxof(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("maxof")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_maxof(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> {
                       var leftIsMax = af.eq(result, left).cast(left.type());
                       var rightIsMax = af.eq(result, right).cast(left.type());
                       return new GradFunction.TensorPair<>(mul(leftIsMax, grads), mul(rightIsMax, grads));
                   })
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> minof(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("minof")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_minof(ptr, left.dereference(), right.dereference(), true))
                   .grads((result, grads) -> {
                       var leftIsMin = af.eq(result, left).cast(left.type());
                       var rightIsMin = af.eq(result, right).cast(left.type());
                       return new GradFunction.TensorPair<>(mul(leftIsMin, grads), mul(rightIsMin, grads));
                   })
                   .build();
    }

    public static <T extends DataType<?, ?>, LD0 extends Num<LD0>, RD0 extends Num<RD0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, N, D1, D2, D3> join(
        Tensor<T, LD0, D1, D2, D3> lhs, Tensor<T, RD0, D1, D2, D3> rhs) {
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(
                       prototype(lhs.type(), shape(n(lhs.d0().size() + rhs.d0().size()), lhs.d1(), lhs.d2(), lhs.d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 0, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(index(grads, seq(lhs.d0())),
                       index(grads, seq(lhs.d0().size(), rhs.d0()))))
                   .build();
    }

    public static <T extends DataType<?, ?>, LD1 extends Num<LD1>, RD1 extends Num<RD1>, D0 extends Num<D0>, D2 extends Num<D2>, D3 extends Num<D3>> Tensor<T, D0, N, D2, D3> join(
        Tensor<T, D0, LD1, D2, D3> lhs, Tensor<T, D0, RD1, D2, D3> rhs, arrayfire.D1 ignored) {
        if (!(lhs.d0().size() == rhs.d0().size() && lhs.d2().size() == rhs.d2().size() &&
                  lhs.d3().size() == rhs.d3().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d1: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(
                       prototype(lhs.type(), shape(lhs.d0(), n(lhs.d1().size() + rhs.d1().size()), lhs.d2(), lhs.d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 1, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(index(grads, span(), seq(lhs.d1())),
                       index(grads, span(), seq(lhs.d1().size(), rhs.d1()))))
                   .build();
    }

    public static <T extends DataType<?, ?>, LD2 extends Num<LD2>, RD2 extends Num<RD2>, D0 extends Num<D0>, D1 extends Num<D1>, D3 extends Num<D3>> Tensor<T, D0, D1, N, D3> join(
        Tensor<T, D0, D1, LD2, D3> lhs, Tensor<T, D0, D1, RD2, D3> rhs, arrayfire.D2 ignored) {
        if (!(lhs.d0().size() == rhs.d0().size() && lhs.d1().size() == rhs.d1().size() &&
                  lhs.d3().size() == rhs.d3().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d2: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(
                       prototype(lhs.type(), shape(lhs.d0(), lhs.d1(), n(lhs.d2().size() + rhs.d2().size()), lhs.d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 2, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(index(grads, span(), span(), seq(lhs.d2())),
                       index(grads, span(), span(), seq(lhs.d2().size(), rhs.d2()))))
                   .build();
    }

    public static <T extends DataType<?, ?>, LD3 extends Num<LD3>, RD3 extends Num<RD3>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>> Tensor<T, D0, D1, D2, N> join(
        Tensor<T, D0, D1, D2, LD3> lhs, Tensor<T, D0, D1, D2, RD3> rhs, arrayfire.D3 ignored) {
        if (!(lhs.d0().size() == rhs.d0().size() && lhs.d1().size() == rhs.d1().size() &&
                  lhs.d2().size() == rhs.d2().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d3: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(
                       prototype(lhs.type(), shape(lhs.d0(), lhs.d1(), lhs.d2(), n(lhs.d3().size() + rhs.d3().size()))))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 3, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new GradFunction.TensorPair<>(
                       index(grads, span(), span(), span(), seq(lhs.d3())),
                       index(grads, span(), span(), span(), seq(lhs.d3().size(), rhs.d3()))))
                   .build();
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, U, D1, D2, D3> sum(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return sum(tensor, D0);
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, U, D1, D2, D3> sum(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType())
                   .grads((result, grads) -> grads.cast(tensor.type()).tileAs(tensor))
                   .build();

    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, U, D2, D3> sum(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType())
                   .grads((result, grads) -> grads.cast(tensor.type()).tileAs(tensor))
                   .build();
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, D1, U, D3> sum(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType())
                   .grads((result, grads) -> grads.cast(tensor.type()).tileAs(tensor))
                   .build();
    }

    public static <ST extends DataType<?, ?>, T extends DataType<?, ST>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<ST, D0, D1, D2, U> sum(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("sum", tensor, arrayfire_h::af_sum, dim, tensor.type().sumType())
                   .grads((result, grads) -> grads.cast(tensor.type()).tileAs(tensor))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> mean(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return mean(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> mean(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type())
                   .grads((result, grads) -> af.div(grads.tileAs(tensor),
                       af.constant(tensor.type(), tensor.d0().size()).tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> mean(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type())
                   .grads((result, grads) -> af.div(grads.tileAs(tensor),
                       af.constant(tensor.type(), tensor.d1().size()).tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> mean(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type())
                   .grads((result, grads) -> af.div(grads.tileAs(tensor),
                       af.constant(tensor.type(), tensor.d2().size()).tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> mean(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("mean", tensor, arrayfire_h::af_mean, dim, tensor.type())
                   .grads((result, grads) -> af.div(grads.tileAs(tensor),
                       af.constant(tensor.type(), tensor.d3().size()).tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> median(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return median(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> median(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> median(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> median(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> median(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("median", tensor, arrayfire_h::af_median, dim, tensor.type()).build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> max(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return max(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> max(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> max(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> max(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> max(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("max", tensor, arrayfire_h::af_max, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> min(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return min(tensor, D0);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, U, D1, D2, D3> min(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D0 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, U, D2, D3> min(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D1 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, U, D3> min(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D2 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, U> min(
        Tensor<T, D0, D1, D2, D3> tensor, arrayfire.D3 dim) {
        return reduce("min", tensor, arrayfire_h::af_min, dim, tensor.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(tensor), tensor).cast(grads.type()),
                       grads.tileAs(tensor)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> ImaxResult<T, U, D1, D2, D3> imax(
        Tensor<T, D0, D1, D2, D3> tensor) {
        var shape = shape(u(), tensor.d1(), tensor.d2(), tensor.d3());
        var pair = operation("imax")
                       .inputs(tensor)
                       .outputs(prototype(tensor.type(), shape), prototype(U32, shape))
                       .operation(
                           (leftPtr, rightPtr) -> arrayfire_h.af_imax(leftPtr, rightPtr, tensor.dereference(), 0))
                       .build();
        return new ImaxResult<>(pair.left(), pair.right());
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, K extends Num<?>> TopKResult<T, K, D1, D2, D3> topk(
        Tensor<T, D0, D1, D2, D3> tensor, K k) {
        var shape = shape(k, tensor.d1(), tensor.d2(), tensor.d3());
        var pair = operation("topk")
                       .inputs(tensor)
                       .outputs(prototype(tensor.type(), shape), prototype(U32, shape))
                       .operation(
                           (leftPtr, rightPtr) -> arrayfire_h.af_topk(leftPtr, rightPtr, tensor.dereference(), k.size(),
                               0, 0))
                       .build();
        return new TopKResult<>(pair.left(), pair.right());
    }

    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D0, D2, D3> diag(
        Tensor<T, D0, U, D2, D3> tensor) {
        return operation("diag")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), shape(tensor.d0(), tensor.d0(), tensor.d2(), tensor.d3())))
                   .operation(ptr -> arrayfire_h.af_diag_create(ptr, tensor.dereference(), 0))
                   // TODO: Implement grad function.
                   .build();
    }

    // https://arrayfire.org/docs/group__blas__func__matmul.htm
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>, OD1 extends Num<?>> Tensor<T, D0, OD1, D2, D3> matmul(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D1, OD1, D2, D3> right) {
        if (left.d1().size() != right.d0().size()) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes for matmul, left: %s right: %s", left.shape(), right.shape()));
        }
        return operation("matmul")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), shape(left.d0(), right.d1(), left.d2(), left.d3())))
                   .operation(ptr -> arrayfire_h.af_matmul(ptr, left.dereference(), right.dereference(), 0, 0))
                   .grads((result, grads) -> {
                       var leftGrads = matmul(grads, right.transpose());
                       var rightGrads = matmul(left.transpose(), grads);
                       return new GradFunction.TensorPair<>(leftGrads, rightGrads);
                   })
                   .build();
    }

    public static <T extends DataType<?, ?>, AD0 extends Num<?>, AD1 extends Num<?>, BD1 extends Num<?>, CD1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, AD0, CD1, D2, D3> matmul(
        Tensor<T, AD0, AD1, D2, D3> a, Tensor<T, AD1, BD1, D2, D3> b, Tensor<T, BD1, CD1, D2, D3> c) {
        if (a.d0().size() * b.d1().size() < b.d0().size() * c.d1().size()) {
            var tmp = matmul(a, b);
            var result = matmul(tmp, c);
            tmp.release();
            return result;
        } else {
            var tmp = matmul(b, c);
            var result = matmul(a, tmp);
            tmp.release();
            return result;
        }
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> clamp(
        Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> lo, Tensor<T, D0, D1, D2, D3> hi) {
        return operation("clamp")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(
                       ptr -> arrayfire_h.af_clamp(ptr, tensor.dereference(), lo.dereference(), hi.dereference(), true))
                   .grads((result, grads) -> {
                       var loMask = ge(tensor, lo);
                       var hiMask = le(tensor, hi);
                       return mul(grads, and(loMask, hiMask).cast(grads.type()));
                   })
                   .build();

    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> relu(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return clamp(tensor, constant(tensor.type(), 0f).tileAs(tensor),
            constant(tensor.type(), Double.POSITIVE_INFINITY).tileAs(tensor));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<B8, D0, D1, D2, D3> eq(
        Tensor<T, D0, D1, D2, D3> left, Tensor<T, D0, D1, D2, D3> right) {
        return operation("eq")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_eq(ptr, left.dereference(), right.dereference(), true))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> negate(
        Tensor<T, D0, D1, D2, D3> tensor) {
        var minusOne = constant(tensor.type(), tensor.shape(), -1);
        return mul(tensor, minusOne);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> exp(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("exp")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_exp(ptr, tensor.dereference()))
                   .grads((result, grads) -> mul(grads, result))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> pow(
        Tensor<T, D0, D1, D2, D3> tensor, double pow) {
        return pow(tensor, constant(tensor.type(), tensor.shape(), pow));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> pow(
        Tensor<T, D0, D1, D2, D3> tensor, Tensor<T, D0, D1, D2, D3> pow) {
        return operation("pow")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_pow(ptr, tensor.dereference(), pow.dereference(), false))
                   .grads((result, grads) -> mul(mul(grads, pow),
                       pow(tensor, sub(pow, constant(pow.type(), pow.shape(), 1)))))
                   .build();
    }

    /**
     * Returns 1 for negative numbers and 0 for positive numbers.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> signbit(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("signbit")
                   .inputs(tensor)
                   .outputs(tensor.prototype())
                   .operation(ptr -> arrayfire_h.af_sign(ptr, tensor.dereference()))
                   .build();
    }

    /**
     * Returns -1 for negative numbers and 1 for positive numbers.
     */
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> signum(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("signum")
                   .inputs(tensor)
                   .outputs(tensor.prototype())
                   .operation(tidyOperation(() -> sub(af.constant(tensor.type(), tensor.shape(), 1),
                       mul(af.constant(tensor.type(), tensor.shape(), 2), signbit(tensor)))))
                   .build();
    }

    public static Operation.Builder operation(String name) {
        return new Operation.Builder().name(name);
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> log(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("log")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_log(ptr, tensor.dereference()))
                   .grads((result, grads) -> div(grads, tensor))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> abs(
        Tensor<T, D0, D1, D2, D3> input) {
        return operation("abs")
                   .inputs(input)
                   .outputs(prototype(input))
                   .operation(ptr -> arrayfire_h.af_abs(ptr, input.dereference()))
                   .grads((result, grads) -> mul(grads, signum(input)))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sqrt(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return operation("sqrt")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_sqrt(ptr, tensor.dereference()))
                   .grads((result, grads) -> div(grads, mul(constant(tensor.type(), tensor.shape(), 2), result)))
                   .build();
    }

    public static <T extends DataType<?, T>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> softmax(
        Tensor<T, D0, D1, D2, D3> tensor) {
        return softmax(tensor, 1f);
    }

    public static Function<MemorySegment, Integer> tidyOperation(Supplier<Tensor<?, ?, ?, ?, ?>> fn) {
        return ptr -> {
            var result = tidy(fn);
            ptr.copyFrom(result.segment());
            Scope.untrack(result);
            return Status.AF_SUCCESS.code();
        };
    }

    public static <T extends DataType<?, T>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> softmax(
        Tensor<T, D0, D1, D2, D3> tensor, float temperature) {
        return operation("softmax").inputs(tensor).outputs(prototype(tensor)).operation(tidyOperation(() -> {
            var max = max(tensor);
            var normalized = sub(tensor, max.tileAs(tensor));
            var exp = exp(div(normalized, constant(tensor.type(), tensor.shape(), temperature)));
            return div(exp, sum(exp).tileAs(tensor));
        })).grads((result, grads) -> {
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
        }).build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sigmoid(
        Tensor<T, D0, D1, D2, D3> tensor) {
        var one = ones(tensor);
        return div(one, add(one, exp(negate(tensor))));
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, D2 extends Num<?>, D3 extends Num<?>> Tensor<T, D0, D1, D2, D3> sparse(
        Tensor<T, D0, D1, D2, D3> tensor, Storage storage) {
        return operation("sparse")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), tensor.shape()))
                   .operation(
                       ptr -> arrayfire_h.af_create_sparse_array_from_dense(ptr, tensor.dereference(), storage.code()))
                   .grads((result, grads) -> grads)
                   .build();
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
        return operation("index")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(),
                       shape(i0.createDim(i0.size()), i1.createDim(i1.size()), i2.createDim(i2.size()),
                           i3.createDim(i3.size()))))
                   .operation(ptr -> {
                       var layout = MemoryLayout.sequenceLayout(4, Index.LAYOUT);
                       var nativeIndexes = Arena.ofAuto().allocateArray(Index.LAYOUT, 4);
                       i0.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(0)),
                           Index.LAYOUT.byteSize()));
                       i1.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(1)),
                           Index.LAYOUT.byteSize()));
                       i2.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(2)),
                           Index.LAYOUT.byteSize()));
                       i3.emigrate(nativeIndexes.asSlice(layout.byteOffset(MemoryLayout.PathElement.sequenceElement(3)),
                           Index.LAYOUT.byteSize()));
                       return arrayfire_h.af_index_gen(ptr, tensor.dereference(), 4, nativeIndexes);
                   })
                   // TODO: Add grads once I work out how to invert and index.
                   .build();

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
        if (newShape.capacity() % tensor.shape().capacity() != 0) {
            throw new IllegalArgumentException(
                String.format("Can't tile perfectly from %s to %s", tensor.shape(), newShape));
        }
        int d0ratio = newShape.d0().size() / tensor.d0().size();
        int d1ratio = newShape.d1().size() / tensor.d1().size();
        int d2ratio = newShape.d2().size() / tensor.d2().size();
        int d3ratio = newShape.d3().size() / tensor.d3().size();
        return operation("tile")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), newShape))
                   .operation(ptr -> arrayfire_h.af_tile(ptr, tensor.dereference(), d0ratio, d1ratio, d2ratio, d3ratio))
                   .grads((result, grads) -> sumAs((Tensor) grads, tensor.shape()).cast(tensor.type()))
                   .build();
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
        return operation("flip")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_flip(ptr, tensor.dereference(), 0))
                   .grads((result, grads) -> flip(grads))
                   .build();
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
        // TODO: CoPilot wrote this, needs tests.
        var computedShape = shape(
            n((tensor.d0().size() + 2 * padding.d0().size() - (filters.d0().size() - 1) * dilation.d0().size() - 1) /
                  stride.d0().size() + 1),
            n((tensor.d1().size() + 2 * padding.d1().size() - (filters.d1().size() - 1) * dilation.d1().size() - 1) /
                  stride.d1().size() + 1), filters.d3(), tensor.d3());
        return operation("convolve2")
                   .inputs(tensor, filters)
                   .outputs(prototype(tensor.type(), computedShape))
                   .operation(ptr -> {
                       retryWithGc(() -> handleStatus(
                           () -> arrayfire_h.af_convolve2_nn(ptr, tensor.dereference(), filters.dereference(), 2,
                               nativeDims(stride), 2, nativeDims(padding), 2, nativeDims(dilation))));
                       return Status.AF_SUCCESS.code();
                   })
                   .build();
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
        var trio = operation("svd")
                       .inputs(tensor)
                       .outputs(prototype(tensor.type(), shape(tensor.d0(), tensor.d0())),
                           prototype(tensor.type(), shape(tensor.d0())),
                           prototype(tensor.type(), shape(tensor.d1(), tensor.d1())))
                       .operation((u, s, v) -> arrayfire_h.af_svd(u, s, v, tensor.dereference()))
                       .build();
        return new SvdResult<>(trio.left(), trio.middle(), trio.right());
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
    public static <T extends DataType<?, ?>, D0 extends Num<D0>, D1 extends Num<?>> Tensor<T, D0, D0, U, U> zca(
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
        return operation("inverse")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(ptr -> arrayfire_h.af_inverse(ptr, tensor.dereference(), 0))
                   .build();
    }

    // TODO: Add uncropped version.
    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>> Tensor<T, D0, D1, U, U> rotate(
        Tensor<T, D0, D1, U, U> tensor, float angle, InterpolationType interpolationType) {
        return operation("rotate")
                   .inputs(tensor)
                   .outputs(prototype(tensor))
                   .operation(
                       ptr -> arrayfire_h.af_rotate(ptr, tensor.dereference(), angle, true, interpolationType.code()))
                   .grads((result, grads) -> rotate(grads, -angle, interpolationType))
                   .build();
    }

    public static <T extends DataType<?, ?>, D0 extends Num<?>, D1 extends Num<?>, ND0 extends Num<?>, ND1 extends Num<?>> Tensor<T, ND0, ND1, U, U> scale(
        Tensor<T, D0, D1, U, U> tensor, ND0 nd0, ND1 nd1, InterpolationType interpolationType) {
        return operation("scale")
                   .inputs(tensor)
                   .outputs(prototype(tensor.type(), shape(nd0, nd1)))
                   .operation(
                       ptr -> arrayfire_h.af_scale(ptr, tensor.dereference(), (float) nd0.size() / tensor.d0().size(),
                           (float) nd1.size() / tensor.d1().size(), nd0.size(), nd1.size(), interpolationType.code()))
                   .grads((result, grads) -> scale(grads, tensor.d0(), tensor.d1(), interpolationType))
                   .build();
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

    public static U u() {
        return U;
    }

    public static U u(int value) {
        return U;
    }

    public static <T extends Tensor<?, ?, ?, ?, ?>> T grads(Tensor<?, ?, ?, ?, ?> loss, T tensor) {
        var graph = new Graph(scope().operations());
        return graph.grads(loss, tensor);
    }

    public static void optimize(Tensor<?, ?, ?, ?, ?> loss) {
        var graph = new Graph(scope().operations());
        graph.optimize(loss);
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

    static void handleStatus(Supplier<Object> res) {
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
}
