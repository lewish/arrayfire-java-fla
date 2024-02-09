package arrayfire;

import arrayfire.capi.arrayfire_h;
import arrayfire.numbers.*;
import arrayfire.optimizers.OptimizerProvider;
import arrayfire.utils.Functions;
import arrayfire.utils.Reference;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;

public class ArrayFire {

    public static final U8 U8 = new U8();
    public static final U64 U64 = new U64();
    public static final S64 S64 = new S64();
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
     * Sorts a array over D0.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sort(Array<T, S> array) {
        return sort(array, D0);
    }

    /**
     * Sorts a array over the given dimension.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sort(Array<T, S> array, Dim dim) {
        return sort(array, dim, true);
    }

    /**
     * Sorts a array over the given dimension in ascending or descending order.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sort(Array<T, S> array, Dim dim,
                                                                                        boolean ascending) {
        return operation("sort")
                   .inputs(array)
                   .outputs(array.prototype())
                   .operation(ptr -> arrayfire_h.af_sort(ptr, array.dereference(), dim.index(), ascending))
                   .build();
    }

    /**
     * Returns a prototype array with the given type and shape.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Prototype<T, S> prototype(T type, S shape) {
        return new Prototype<>(type, shape);
    }

    /**
     * Returns a prototype array with the same type and shape as the given array.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Prototype<T, S> prototype(Array<T, S> array) {
        return new Prototype<>(array.type(), array.shape());
    }

    /**
     * Sorts a array over D0 and returns the values and indices of original values.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> SortIndexResult<T, S> sortIndex(
        Array<T, S> array) {
        return sortIndex(array, D0);
    }

    /**
     * Sorts a array over the given dimension and returns the values and indices of original values.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> SortIndexResult<T, S> sortIndex(
        Array<T, S> array, Dim dim) {
        return sortIndex(array, dim, true);
    }

    /**
     * Sorts a array over the given dimension in ascending or descending order and returns the values and indices of original values.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> SortIndexResult<T, S> sortIndex(
        Array<T, S> array, Dim dim, boolean ascending) {
        var pair = operation("sort_index")
                       .inputs(array)
                       .outputs(prototype(array.type(), array.shape()), prototype(U32, array.shape()))
                       .operation(
                           (leftPtr, rightPtr) -> arrayfire_h.af_sort_index(leftPtr, rightPtr, array.dereference(),
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
     * Creates a device array from the given native array and shape.
     */
    public static <DT extends DataType<? extends DataType.Meta<?, ?, ?>>, S extends Shape<?, ?, ?, ?>, HA extends HostArray<DT, ?, S>> Array<DT, S> create(
        HA array) {
        return operation("create")
                   .inputs()
                   .outputs(prototype(array.type(), array.shape()))
                   .operation(ptr -> arrayfire_h.af_create_array(ptr, array.segment(), array.shape().ndims(),
                       nativeDims(array.shape()), array.type().code()))
                   .build();
    }

    /**
     * Creates a device array from the given type and java values.
     * This is not recommended in a production setting, as memory will be copied twice. Instead, use {@link #create(HostArray)}.
     */
    @SafeVarargs
    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>> Array<DT, Shape<N, U, U, U>> create(
        DT type, JT... values) {
        return tidy(() -> {
            var array = createHost(type, shape(values.length), false);
            for (int i = 0; i < values.length; i++) {
                array.set(i, values[i]);
            }
            return create(array);
        });
    }

    /**
     * Creates a device array from the given type and java native array.
     * This is not recommended in a production setting, as memory will be copied twice. Instead, use {@link #create(HostArray)}.
     */
    @SuppressWarnings("unchecked")
    public static <JTA, JT, DTM extends DataType.Meta<?, JT, JTA>, DT extends DataType<DTM>> Array<DT, Shape<N, U, U, U>> create(
        DT type, JTA values) {
        return tidy(() -> {
            var length = java.lang.reflect.Array.getLength(values);
            var array = createHost(type, shape(length), false);
            for (int i = 0; i < length; i++) {
                array.set(i, (JT) java.lang.reflect.Array.get(values, i));
            }
            return create(array);
        });
    }

    /**
     * Creates a {@link F32} device array from the given float values.
     */
    public static Array<F32, Shape<N, U, U, U>> create(float... values) {
        return create(F32, values);
    }

    /**
     * Creates a {@link F64} device array from the given double values.
     */
    public static Array<F64, Shape<N, U, U, U>> create(double... values) {
        return create(F64, values);
    }

    /**
     * Creates a {@link S32} device array from the given byte values.
     */
    public static Array<S32, Shape<N, U, U, U>> create(int... values) {
        return create(S32, values);
    }

    /**
     * Creates a constant scalar {@link F32} device array from the given float value.
     */
    public static Array<F32, Shape<U, U, U, U>> constant(float value) {
        return constant(F32, value);
    }

    /**
     * Creates a constant scalar {@link F64} device array from the given float value.
     */
    public static Array<F64, Shape<U, U, U, U>> constant(double value) {
        return constant(F64, value);
    }

    /**
     * Creates a constant scalar {@link U8} device array from the given byte value.
     */
    public static Array<U8, Shape<U, U, U, U>> constant(byte value) {
        return constant(U8, value);
    }

    /**
     * Creates a constant scalar {@link S32} device array from the given int value.
     */
    public static Array<S32, Shape<U, U, U, U>> constant(int value) {
        return constant(S32, value);
    }

    /**
     * Creates a constant scalar {@link S64} device array from the given long value.
     */
    public static Array<S64, Shape<U, U, U, U>> constant(long value) {
        return constant(S64, value);
    }


    /**
     * Creates a constant scalar device array from the given type and double value.
     */
    public static <DT extends DataType<?>> Array<DT, Shape<U, U, U, U>> constant(DT type, double value) {
        return constant(type, scalar(), value);
    }

    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<DT, JT, S> createHost(
        DT type, S shape) {
        return createHost(type, shape, false);
    }

    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<DT, JT, S> createHost(
        DT type, S shape, boolean pinned) {
        var result = new HostArray<>(type, shape, pinned);
        scope().register(result);
        return result;
    }

    @SafeVarargs
    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>> HostArray<DT, JT, Shape<N, U, U, U>> createHost(
        DT type, JT... values) {
        return createHost(type, shape(values.length), false, values);
    }

    @SafeVarargs
    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<DT, JT, S> createHost(
        DT type, S shape, JT... values) {
        return createHost(type, shape, false, values);
    }

    @SafeVarargs
    public static <JT, DTM extends DataType.Meta<?, JT, ?>, DT extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<DT, JT, S> createHost(
        DT type, S shape, boolean pinned, JT... values) {
        var hostArray = createHost(type, shape, pinned);
        for (int i = 0; i < values.length; i++) {
            hostArray.set(i, values[i]);
        }
        return hostArray;
    }

    @SuppressWarnings("unchecked")
    public static <JT, JAT, DTM extends DataType.Meta<?, JT, JAT>, DT extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<DT, JT, S> createHost(
        DT type, S shape, boolean pinned, JAT values) {
        if (shape.capacity() != java.lang.reflect.Array.getLength(values)) {
            throw new IllegalArgumentException(
                String.format("Expected array with capacity of shape %d, but got %d", shape.capacity(),
                    java.lang.reflect.Array.getLength(values)));
        }
        var length = java.lang.reflect.Array.getLength(values);
        var array = createHost(type, shape, pinned);
        for (int i = 0; i < length; i++) {
            array.set(i, (JT) java.lang.reflect.Array.get(values, i));
        }
        return array;
    }

    /**
     * Creates a {@link F32} device array from the given float values.
     */
    public static HostArray<F32, Float, Shape<N, U, U, U>> createHost(float... values) {
        return createHost(F32, shape(values.length), false, values);
    }

    /**
     * Creates a {@link F64} device array from the given double values.
     */
    public static HostArray<F64, Double, Shape<N, U, U, U>> createHost(double... values) {
        return createHost(F64, shape(values.length), false, values);
    }

    /**
     * Creates a {@link S32} device array from the given byte values.
     */
    public static HostArray<S32, Integer, Shape<N, U, U, U>> createHost(int... values) {
        return createHost(S32, shape(values.length), false, values);
    }

    public static Shape<U, U, U, U> scalar() {
        return shape(u());
    }

    /**
     * Creates a constant device array from the given type, shape, and double value.
     */
    public static <DT extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<DT, S> constant(DT type, S shape,
                                                                                              double value) {
        return operation("constant")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_constant(ptr, value, shape.ndims(), nativeDims(shape), type.code()))
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
     * Returns a lookup index using the given array as lookup values (indices).
     */
    public static <DT extends DataType<?>, D0 extends Num<D0>, S extends Shape<D0, U, U, U>> Index<D0> seq(
        Array<DT, S> index) {
        return new Index<>(index, index.shape().d0()::create);
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
    public static <D0 extends Num<D0>> Shape<D0, U, U, U> shape(D0 d0) {
        return new Shape<>(d0, u(), u(), u());
    }

    public static <D0 extends Num<D0>> Shape<D0, N, U, U> shape(D0 d0, int d1) {
        return new Shape<>(d0, n(d1), u(), u());
    }

    public static <D1 extends Num<D1>> Shape<N, D1, U, U> shape(int d0, D1 d1) {
        return new Shape<>(n(d0), d1, u(), u());
    }

    public static Shape<N, N, U, U> shape(int d0, int d1) {
        return new Shape<>(n(d0), n(d1), u(), u());
    }

    public static <D0 extends Num<D0>, D1 extends Num<D1>> Shape<D0, D1, U, U> shape(D0 d0, D1 d1) {
        return new Shape<>(d0, d1, u(), u());
    }

    public static Shape<N, N, N, U> shape(int d0, int d1, int d2) {
        return new Shape<>(n(d0), n(d1), n(d2), u());
    }

    public static Shape<N, N, N, N> shape(int d0, int d1, int d2, int d3) {
        return new Shape<>(n(d0), n(d1), n(d2), n(d3));
    }

    public static <D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>> Shape<D0, D1, D2, U> shape(D0 d0, D1 d1,
                                                                                                          D2 d2) {
        return new Shape<>(d0, d1, d2, u());
    }

    public static <D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Shape<D0, D1, D2, D3> shape(
        D0 d0, D1 d1, D2 d2, D3 d3) {
        return new Shape<>(d0, d1, d2, d3);
    }


    public static <D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Shape<D0, D1, D2, D3> shape(
        Index<D0> d0, Index<D1> d1, Index<D2> d2, Index<D3> d3) {
        return new Shape<>(d0.createDim(), d1.createDim(), d2.createDim(), d3.createDim());
    }

    private static <T extends DataType<?>, IT extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Operation.Builder.Unary<Array<IT, S>>.Single<Array<T, Shape<U, D1, D2, D3>>> reduce(
        String name, Array<IT, S> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
        arrayfire.D0 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(u(), a.shape().d1(), a.shape().d2(), a.shape().d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    private static <T extends DataType<?>, IT extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Operation.Builder.Unary<Array<IT, S>>.Single<Array<T, Shape<D0, U, D2, D3>>> reduce(
        String name, Array<IT, S> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
        arrayfire.D1 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.shape().d0(), u(), a.shape().d2(), a.shape().d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));

    }

    private static <T extends DataType<?>, IT extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Operation.Builder.Unary<Array<IT, S>>.Single<Array<T, Shape<D0, D1, U, D3>>> reduce(
        String name, Array<IT, S> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
        arrayfire.D2 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.shape().d0(), a.shape().d1(), u(), a.shape().d3())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    private static <T extends DataType<?>, IT extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Operation.Builder.Unary<Array<IT, S>>.Single<Array<T, Shape<D0, D1, D2, U>>> reduce(
        String name, Array<IT, S> a, Functions.Function3<MemorySegment, MemorySegment, Integer, Integer> method,
        arrayfire.D3 dim, T resultType) {
        return operation(name)
                   .inputs(a)
                   .outputs(prototype(resultType, shape(a.shape().d0(), a.shape().d1(), a.shape().d2(), u())))
                   .operation(ptr -> method.apply(ptr, a.dereference(), dim.index()));
    }

    /**
     * Cast the given array to the given type.
     */
    @SuppressWarnings("unchecked")
    public static <T extends DataType<?>, OT extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<OT, S> cast(
        Array<T, S> input, OT type) {
        if (input.type().equals(type)) {
            return (Array<OT, S>) input;
        }
        return operation("cast")
                   .inputs(input)
                   .outputs(prototype(type, input.shape()))
                   .operation(ptr -> arrayfire_h.af_cast(ptr, input.dereference(), type.code()))
                   .grads((result, grads) -> cast(grads, input.type()))
                   .build();
    }

    /**
     * Returns a array of value 1 with the same type and shape as the given array.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> ones(Array<T, S> model) {
        return ones(model.type(), model.shape());
    }

    /**
     * Returns a array of value 1 with the given type and shape.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> ones(T type, S shape) {
        return constant(type, shape, 1);
    }

    /**
     * Returns a array of value 0 with the given type and shape.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> zeros(T type, S shape) {
        return constant(type, shape, 0);
    }

    /**
     * Create a random array sampled from uniform distribution between [0, 1].
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> randu(T type, S shape) {
        return operation("randu")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_randu(ptr, shape.ndims(), nativeDims(shape), type.code()))
                   .build();
    }

    /**
     * Create a random array sampled from a normal distribution with mean 0.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> randn(T type, S shape) {
        return operation("randn")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_randn(ptr, shape.ndims(), nativeDims(shape), type.code()))
                   .build();
    }

    /**
     * Create a array with values [0, n-1].
     */
    public static Array<U32, Shape<N, U, U, U>> range(int n) {
        return range(U32, n);
    }

    /**
     * Create a array with values [0, n-1] of the given type.
     */
    public static <T extends DataType<?>> Array<T, Shape<N, U, U, U>> range(T type, int n) {
        var shape = shape(n(n));
        return operation("range")
                   .inputs()
                   .outputs(prototype(type, shape))
                   .operation(ptr -> arrayfire_h.af_range(ptr, shape.ndims(), nativeDims(shape), 0, type.code()))
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
    public static <JT, DTM extends DataType.Meta<?, JT, ?>, T extends DataType<DTM>, S extends Shape<?, ?, ?, ?>> HostArray<T, JT, S> data(
        Array<T, S> a) {
        var result = createHost(a.type(), a.shape(), false);
        handleStatus(() -> arrayfire_h.af_get_data_ptr(result.segment(), a.dereference()));
        return result;
    }

    public static <JAT, DTM extends DataType.Meta<?, ?, JAT>, DT extends DataType<DTM>, HA extends HostArray<DT, ?, ?>> JAT heap(
        HA array) {
        var length = array.length();
        var heapArray = array.type().meta().createHeapArray(array.shape().capacity());
        for (int i = 0; i < length; i++) {
            java.lang.reflect.Array.set(heapArray, i, array.get(i));
        }
        return heapArray;
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
     * Transpose D0 and D1 dimensions of the given array.
     */
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D1, D0, D2, D3>> transpose(
        Array<T, S> array) {
        return operation("transpose")
                   .inputs(array)
                   .outputs(prototype(array.type(),
                       shape(array.shape().d1(), array.shape().d0(), array.shape().d2(), array.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_transpose(ptr, array.dereference(), true))
                   .grads((result, grads) -> transpose(grads).reshape(array.shape()))
                   .build();
    }

    /**
     * Change the type of the array's D0 dimension to the given type variable provider.
     */
    public static <T extends DataType<?>, OD0 extends Num<OD0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<OD0, D1, D2, D3>> castshape(
        Array<T, ? extends Shape<?, D1, D2, D3>> array, Function<Integer, OD0> d0) {
        return reshape(array,
            shape(d0.apply(array.shape().d0().size()), array.shape().d1(), array.shape().d2(), array.shape().d3()));
    }

    /**
     * Change the type of the array's D0, D1 dimensions to the given type variable providers.
     */
    public static <T extends DataType<?>, OD0 extends Num<OD0>, OD1 extends Num<OD1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<OD0, OD1, D2, D3>> castshape(
        Array<T, Shape<?, ?, D2, D3>> array, Function<Integer, OD0> d0, Function<Integer, OD1> d1) {
        return reshape(array,
            shape(d0.apply(array.shape().d0().size()), d1.apply(array.shape().d1().size()), array.shape().d2(),
                array.shape().d3()));
    }

    /**
     * Change the type of the array's D0, D1, D2 dimensions to the given type variable providers.
     */
    public static <T extends DataType<?>, OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>, D3 extends Num<D3>> Array<T, Shape<OD0, OD1, OD2, D3>> castshape(
        Array<T, Shape<?, ?, ?, D3>> array, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
        Function<Integer, OD2> d2) {
        return reshape(array, shape(d0.apply(array.shape().d0().size()), d1.apply(array.shape().d1().size()),
            d2.apply(array.shape().d2().size()), array.shape().d3()));
    }

    /**
     * Change the type of the array's dimensions to the given type variable providers.
     */
    public static <T extends DataType<?>, OD0 extends Num<OD0>, OD1 extends Num<OD1>, OD2 extends Num<OD2>, OD3 extends Num<OD3>> Array<T, Shape<OD0, OD1, OD2, OD3>> castshape(
        Array<T, ? extends Shape<?, ?, ?, ?>> array, Function<Integer, OD0> d0, Function<Integer, OD1> d1,
        Function<Integer, OD2> d2, Function<Integer, OD3> d3) {
        return reshape(array, shape(d0.apply(array.shape().d0().size()), d1.apply(array.shape().d1().size()),
            d2.apply(array.shape().d2().size()), d3.apply(array.shape().d3().size())));
    }

    /**
     * Reshape the array to the given shape.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, NS extends Shape<?, ?, ?, ?>> Array<T, NS> reshape(
        Array<T, S> array, NS newShape) {
        if (array.shape().capacity() != newShape.capacity()) {
            throw new IllegalArgumentException(
                String.format("New shape %s doesn't have same capacity as original shape %s", newShape, array.shape()));
        }
        return operation("reshape")
                   .inputs(array)
                   .outputs(prototype(array.type(), newShape))
                   .operation(
                       ptr -> arrayfire_h.af_moddims(ptr, array.dereference(), newShape.ndims(), nativeDims(newShape)))
                   .grads((result, grads) -> reshape(grads, array.shape()))
                   .build();
    }

    /**
     * Release the memory of the given array on the device.
     */
    public static void release(Array<?, ?> array) {
        handleStatus(() -> arrayfire_h.af_release_array(array.dereference()));
        Scope.untrack(array);
    }

    /**
     * Retain the given array, increasing its ref count by 1 and return a new container for it.
     */
    public static <DT extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<DT, S> retain(Array<DT, S> array) {
        return operation("retain")
                   .inputs(array)
                   .outputs(prototype(array.type(), array.shape()))
                   .operation(ptr -> arrayfire_h.af_retain_array(ptr, array.dereference()))
                   .grads((result, grads) -> grads)
                   .build();
    }

    /**
     * Set the values of the given variable to the values of the given array.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Operation set(Variable<T, S> variable,
                                                                                     Array<T, S> array) {
        return operation("set").inputs(array).outputs().operation(() -> {
            handleStatus(() -> arrayfire_h.af_release_array(variable.dereference()));
            handleStatus(() -> arrayfire_h.af_retain_array(variable.segment(), array.dereference()));
        }).build();
    }

    /**
     * Return the ref count of the given array.
     */
    public static int refCount(Array<?, ?> array) {
        try (Arena arena = Arena.ofConfined()) {
            var result = arena.allocate(ValueLayout.JAVA_INT);
            handleStatus(() -> arrayfire_h.af_get_data_ref_count(result, array.dereference()));
            return result.get(ValueLayout.JAVA_INT, 0);
        }
    }

    /**
     * Create a variable with the given initializer.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Variable<T, S> variable(
        Supplier<Array<T, S>> initializer) {
        var tensor = af.tidy(initializer);
        var variable = new Variable<>(tensor.type(), tensor.shape());
        variable.segment().copyFrom(tensor.segment());
        Scope.untrack(tensor);
        return variable;
    }

    /**
     * Create params with the given initializer and optimizer.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Params<T, S> params(
        Supplier<Array<T, S>> initializer, OptimizerProvider optimizerProvider) {
        var tensor = af.tidy(initializer);
        var params = new Params<>(tensor.type(), tensor.shape(), optimizerProvider);
        params.segment().copyFrom(tensor.segment());
        Scope.untrack(tensor);
        return params;
    }

    /**
     * Evaluate the array, telling the ArrayFire JIT compiler that you want the literal values of the array.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> eval(Array<T, S> array) {
        handleStatus(() -> arrayfire_h.af_eval(array.dereference()));
        return array;
    }

    /**
     * Evaluate the arrays, telling the ArrayFire JIT compiler that you want the literal values of the arrays.
     */
    public static void eval(Array<?, ?>... arrays) {
        try (Arena arena = Arena.ofConfined()) {
            var array = arena.allocateArray(ValueLayout.ADDRESS, arrays.length);
            for (int i = 0; i < arrays.length; i++) {
                array.setAtIndex(ValueLayout.ADDRESS, i, arrays[i].dereference());
            }
            handleStatus(() -> arrayfire_h.af_eval_multiple(arrays.length, array));
        }
    }

    private static void assertTileable(Tileable<?, ?> tileable, Array<?, ?> array) {
        assertTileable(array, tileable);
    }

    private static void assertTileable(Array<?, ?> array, Tileable<?, ?> tileable) {
        if (array.shape().d0().size() < tileable.array().shape().d0().size() ||
                array.shape().d1().size() < tileable.array().shape().d1().size() ||
                array.shape().d2().size() < tileable.array().shape().d2().size() ||
                array.shape().d3().size() < tileable.array().shape().d3().size()) {
            throw new IllegalArgumentException(
                String.format("Tileable shape %s is larger than array shape %s", tileable.array().shape(),
                    array.shape()));
        }
    }

    private static void assertShapeEquals(Shape<?, ?, ?, ?> left, Shape<?, ?, ?, ?> right) {
        if (!Arrays.equals(left.dims(), right.dims())) {
            throw new IllegalArgumentException(String.format("Shapes %s and %s are not equal", left, right));
        }
    }

    /**
     * Multiply the array by a scalar value.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> mul(Array<T, S> left, double right) {
        return mul(left, af.constant(left.type(), left.shape(), right));
    }

    /**
     * Multiply two tensors together element wise, broadcasting the smaller array to the larger array's shape.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> mul(Array<T, S> array,
                                                                                       Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return mul(array, tileable.array().tileAs(array));
    }

    /**
     * Multiply two tensors together, element wise.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> mul(Array<T, S> left,
                                                                                       Array<T, S> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("mul")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_mul(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> new ArrayPair<>(mul(grads, right), mul(grads, left)))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> div(double left, Array<T, S> right) {
        return div(af.constant(right.type(), left).tile(), right);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> div(Array<T, S> left, double right) {
        return div(left, af.constant(left.type(), right).tile());
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> div(Array<T, S> left,
                                                                                       Tileable<T, ?> right) {
        assertTileable(left, right);
        return div(left, right.array().tileAs(left));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> div(Tileable<T, ?> left,
                                                                                       Array<T, S> right) {
        assertTileable(left, right);
        return div(left.array().tileAs(right), right);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> div(Array<T, S> left,
                                                                                       Array<T, S> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("div")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_div(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> {
                       var rightReciprocal = div(constant(1f).cast(left.type()).tileAs(right), right);
                       var leftGrads = mul(rightReciprocal, grads);
                       var rightGrads = mul(mul(leftGrads, left.negate()), rightReciprocal);
                       return new ArrayPair<>(leftGrads, rightGrads);
                   })
                   .build();

    }


    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> add(Array<T, S> left, double right) {
        return add(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> add(Array<T, S> array,
                                                                                       Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return add(array, tileable.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<T, SL> add(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("add")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_add(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> new ArrayPair<>(grads, grads.reshape(right.shape())))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sub(double left, Array<T, S> right) {
        return sub(af.constant(right.type(), left).tile(), right);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sub(Array<T, S> left, double right) {
        return sub(left, af.constant(left.type(), right).tile());
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sub(Array<T, S> left,
                                                                                       Tileable<T, ?> right) {
        assertTileable(left, right);
        return sub(left, right.array().tileAs(left));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sub(Tileable<T, ?> left,
                                                                                       Array<T, S> right) {
        assertTileable(left, right);
        return sub(left.array().tileAs(right), right);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<T, SL> sub(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("sub")
                   .inputs(left, right)
                   .outputs(prototype(left))
                   .operation(ptr -> arrayfire_h.af_sub(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> new ArrayPair<>(grads, grads.negate().reshape(right.shape())))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<B8, S> ge(Array<T, S> left, double right) {
        return ge(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<B8, S> ge(Array<T, S> array,
                                                                                       Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return ge(array, tileable.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<B8, SL> ge(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("ge")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_ge(ptr, left.dereference(), right.dereference(), false))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<B8, S> le(Array<T, S> left, double right) {
        return le(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<B8, S> le(Array<T, S> array,
                                                                                       Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return le(array, tileable.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<B8, SL> le(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("le")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_le(ptr, left.dereference(), right.dereference(), false))
                   .build();
    }

    public static <S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<B8, SL> and(Array<B8, SL> left,
                                                                                              Array<B8, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("and")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_and(ptr, left.dereference(), right.dereference(), false))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> maxof(Array<T, S> left,
                                                                                         double right) {
        return maxof(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> maxof(Array<T, S> array,
                                                                                         Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return maxof(array, tileable.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<T, SL> maxof(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("maxof")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_maxof(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> {
                       var leftIsMax = eq(result, left).cast(left.type());
                       var rightIsMax = eq(result, right).cast(left.type());
                       return new ArrayPair<>(mul(leftIsMax, grads), mul(rightIsMax, grads).reshape(right.shape()));
                   })
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> minof(Array<T, S> left,
                                                                                         double right) {
        return minof(left, af.constant(left.type(), left.shape(), right));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> minof(Array<T, S> array,
                                                                                         Tileable<T, ?> tileable) {
        assertTileable(array, tileable);
        return minof(array, tileable.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<T, SL> minof(
        Array<T, SL> left, Array<T, SR> right) {
        assertShapeEquals(left.shape(), right.shape());
        return operation("minof")
                   .inputs(left, right)
                   .outputs(prototype(left.type(), left.shape()))
                   .operation(ptr -> arrayfire_h.af_minof(ptr, left.dereference(), right.dereference(), false))
                   .grads((result, grads) -> {
                       var leftIsMin = eq(result, left).cast(left.type());
                       var rightIsMin = eq(result, right).cast(left.type());
                       return new ArrayPair<>(mul(leftIsMin, grads), mul(rightIsMin, grads).reshape(right.shape()));
                   })
                   .build();
    }

    public static <T extends DataType<?>, LD0 extends Num<LD0>, RD0 extends Num<RD0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, SL extends Shape<LD0, D1, D2, D3>, SR extends Shape<RD0, D1, D2, D3>> Array<T, Shape<N, D1, D2, D3>> join(
        Array<T, SL> lhs, Array<T, SR> rhs) {
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(prototype(lhs.type(),
                       shape(n(lhs.shape().d0().size() + rhs.shape().d0().size()), lhs.shape().d1(), lhs.shape().d2(),
                           lhs.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 0, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new ArrayPair<>(index(grads, seq(lhs.shape().d0())).reshape(lhs.shape()),
                       index(grads, seq(lhs.shape().d0().size(), rhs.shape().d0())).reshape(rhs.shape())))
                   .build();
    }

    public static <T extends DataType<?>, LD1 extends Num<LD1>, RD1 extends Num<RD1>, D0 extends Num<D0>, D2 extends Num<D2>, D3 extends Num<D3>, SL extends Shape<D0, LD1, D2, D3>, SR extends Shape<D0, RD1, D2, D3>> Array<T, Shape<D0, N, D2, D3>> join(
        Array<T, SL> lhs, Array<T, SR> rhs, arrayfire.D1 ignored) {
        if (!(lhs.shape().d0().size() == rhs.shape().d0().size() &&
                  lhs.shape().d2().size() == rhs.shape().d2().size() &&
                  lhs.shape().d3().size() == rhs.shape().d3().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d1: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(prototype(lhs.type(),
                       shape(lhs.shape().d0(), n(lhs.shape().d1().size() + rhs.shape().d1().size()), lhs.shape().d2(),
                           lhs.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 1, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new ArrayPair<>(
                       index(grads, span(), seq(lhs.shape().d1())).reshape(lhs.shape()),
                       index(grads, span(), seq(lhs.shape().d1().size(), rhs.shape().d1())).reshape(rhs.shape())))
                   .build();
    }

    public static <T extends DataType<?>, LD2 extends Num<LD2>, RD2 extends Num<RD2>, D0 extends Num<D0>, D1 extends Num<D1>, D3 extends Num<D3>, SL extends Shape<D0, D1, LD2, D3>, SR extends Shape<D0, D1, RD2, D3>> Array<T, Shape<D0, D1, N, D3>> join(
        Array<T, SL> lhs, Array<T, SR> rhs, arrayfire.D2 ignored) {
        if (!(lhs.shape().d0().size() == rhs.shape().d0().size() &&
                  lhs.shape().d1().size() == rhs.shape().d1().size() &&
                  lhs.shape().d3().size() == rhs.shape().d3().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d2: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(prototype(lhs.type(),
                       shape(lhs.shape().d0(), lhs.shape().d1(), n(lhs.shape().d2().size() + rhs.shape().d2().size()),
                           lhs.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 2, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new ArrayPair<>(
                       index(grads, span(), span(), seq(lhs.shape().d2())).reshape(lhs.shape()),
                       index(grads, span(), span(), seq(lhs.shape().d2().size(), rhs.shape().d2())).reshape(
                           rhs.shape())))
                   .build();
    }

    public static <T extends DataType<?>, LD3 extends Num<LD3>, RD3 extends Num<RD3>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, SL extends Shape<D0, D1, D2, LD3>, SR extends Shape<D0, D1, D2, RD3>> Array<T, Shape<D0, D1, D2, N>> join(
        Array<T, SL> lhs, Array<T, SR> rhs, arrayfire.D3 ignored) {
        if (!(lhs.shape().d0().size() == rhs.shape().d0().size() &&
                  lhs.shape().d1().size() == rhs.shape().d1().size() &&
                  lhs.shape().d2().size() == rhs.shape().d2().size())) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes to join along d3: %s, %s", lhs.shape(), rhs.shape()));
        }
        return operation("join")
                   .inputs(lhs, rhs)
                   .outputs(prototype(lhs.type(), shape(lhs.shape().d0(), lhs.shape().d1(), lhs.shape().d2(),
                       n(lhs.shape().d3().size() + rhs.shape().d3().size()))))
                   .operation(ptr -> arrayfire_h.af_join(ptr, 3, lhs.dereference(), rhs.dereference()))
                   .grads((result, grads) -> new ArrayPair<>(
                       index(grads, span(), span(), span(), seq(lhs.shape().d3())).reshape(lhs.shape()),
                       index(grads, span(), span(), span(), seq(lhs.shape().d3().size(), rhs.shape().d3())).reshape(
                           rhs.shape())))
                   .build();
    }

    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<U, D1, D2, D3>> sum(
        Array<T, S> array) {
        return sum(array, D0);
    }

    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<U, D1, D2, D3>> sum(
        Array<T, S> array, arrayfire.D0 dim) {
        return reduce("sum", array, arrayfire_h::af_sum, dim, array.type().meta().sumType())
                   .grads((result, grads) -> grads.cast(array.type()).tileAs(array))
                   .build();

    }

    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<D0, U, D2, D3>> sum(
        Array<T, S> array, arrayfire.D1 dim) {
        return reduce("sum", array, arrayfire_h::af_sum, dim, array.type().meta().sumType())
                   .grads((result, grads) -> grads.cast(array.type()).tileAs(array))
                   .build();
    }

    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<D0, D1, U, D3>> sum(
        Array<T, S> array, arrayfire.D2 dim) {
        return reduce("sum", array, arrayfire_h::af_sum, dim, array.type().meta().sumType())
                   .grads((result, grads) -> grads.cast(array.type()).tileAs(array))
                   .build();
    }

    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<D0, D1, D2, U>> sum(
        Array<T, S> array, arrayfire.D3 dim) {
        return reduce("sum", array, arrayfire_h::af_sum, dim, array.type().meta().sumType())
                   .grads((result, grads) -> grads.cast(array.type()).tileAs(array))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> mean(
        Array<T, S> array) {
        return mean(array, D0);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> mean(
        Array<T, S> array, arrayfire.D0 dim) {
        return reduce("mean", array, arrayfire_h::af_mean, dim, array.type())
                   .grads((result, grads) -> af.div(grads.tileAs(array),
                       af.constant(array.type(), array.shape().d0().size()).tile()))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, U, D2, D3>> mean(
        Array<T, S> array, arrayfire.D1 dim) {
        return reduce("mean", array, arrayfire_h::af_mean, dim, array.type())
                   .grads((result, grads) -> af.div(grads.tileAs(array),
                       af.constant(array.type(), array.shape().d1().size()).tile()))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, U, D3>> mean(
        Array<T, S> array, arrayfire.D2 dim) {
        return reduce("mean", array, arrayfire_h::af_mean, dim, array.type())
                   .grads((result, grads) -> af.div(grads.tileAs(array),
                       af.constant(array.type(), array.shape().d2().size()).tile()))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, D2, U>> mean(
        Array<T, S> array, arrayfire.D3 dim) {
        return reduce("mean", array, arrayfire_h::af_mean, dim, array.type())
                   .grads((result, grads) -> af.div(grads.tileAs(array),
                       af.constant(array.type(), array.shape().d3().size()).tile()))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> median(
        Array<T, S> array) {
        return median(array, D0);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> median(
        Array<T, S> array, arrayfire.D0 dim) {
        return reduce("median", array, arrayfire_h::af_median, dim, array.type()).build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, U, D2, D3>> median(
        Array<T, S> array, arrayfire.D1 dim) {
        return reduce("median", array, arrayfire_h::af_median, dim, array.type()).build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, U, D3>> median(
        Array<T, S> array, arrayfire.D2 dim) {
        return reduce("median", array, arrayfire_h::af_median, dim, array.type()).build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, D2, U>> median(
        Array<T, S> array, arrayfire.D3 dim) {
        return reduce("median", array, arrayfire_h::af_median, dim, array.type()).build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> max(
        Array<T, S> array) {
        return max(array, D0);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> max(
        Array<T, S> array, arrayfire.D0 dim) {
        return reduce("max", array, arrayfire_h::af_max, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, U, D2, D3>> max(
        Array<T, S> array, arrayfire.D1 dim) {
        return reduce("max", array, arrayfire_h::af_max, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, U, D3>> max(
        Array<T, S> array, arrayfire.D2 dim) {
        return reduce("max", array, arrayfire_h::af_max, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, D2, U>> max(
        Array<T, S> array, arrayfire.D3 dim) {
        return reduce("max", array, arrayfire_h::af_max, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> min(
        Array<T, S> array) {
        return min(array, D0);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<U, D1, D2, D3>> min(
        Array<T, S> array, arrayfire.D0 dim) {
        return reduce("min", array, arrayfire_h::af_min, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, U, D2, D3>> min(
        Array<T, S> array, arrayfire.D1 dim) {
        return reduce("min", array, arrayfire_h::af_min, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, U, D3>> min(
        Array<T, S> array, arrayfire.D2 dim) {
        return reduce("min", array, arrayfire_h::af_min, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, Shape<D0, D1, D2, U>> min(
        Array<T, S> array, arrayfire.D3 dim) {
        return reduce("min", array, arrayfire_h::af_min, dim, array.type())
                   .grads((result, grads) -> mul(af.eq(result.tileAs(array), array).cast(grads.type()),
                       grads.tileAs(array)))
                   .build();
    }

    public static <T extends DataType<?>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<?, D1, D2, D3>> ImaxResult<T, Shape<U, D1, D2, D3>> imax(
        Array<T, S> array) {
        var shape = shape(u(), array.shape().d1(), array.shape().d2(), array.shape().d3());
        var pair = operation("imax")
                       .inputs(array)
                       .outputs(prototype(array.type(), shape), prototype(U32, shape))
                       .operation((leftPtr, rightPtr) -> arrayfire_h.af_imax(leftPtr, rightPtr, array.dereference(), 0))
                       .grads(
                           (results, grads) -> mul(af.eq(results.left().tileAs(array), array).cast(grads.left().type()),
                               grads.left().tileAs(array)))
                       .build();
        return new ImaxResult<>(pair.left(), pair.right());
    }

    // create_sparse_array
    public static <T extends DataType<?>, D0 extends Num<D0>, S extends Shape<?, ?, U, U>> Array<T, S> sparse(
        Array<T, ? extends Shape<?, U, U, U>> values, Array<S32, ? extends Shape<D0, U, U, U>> d0Indices,
        Array<S32, ? extends Shape<D0, U, U, U>> d1Indices, S shape) {
        return operation("sparse")
                   .inputs(values)
                   .outputs(prototype(values.type(), shape))
                   .operation(ptr -> arrayfire_h.af_create_sparse_array(ptr, shape.d0().size(), shape.d1().size(),
                       values.dereference(), d0Indices.dereference(), d1Indices.dereference(), Storage.COO.code()))
                   .build();
    }

    public static <T extends DataType<?>, D1 extends Num<D1>, K extends Num<K>, S extends Shape<?, D1, U, U>> TopKResult<T, Shape<K, D1, U, U>> topk(
        Array<T, S> array, K k) {
        var shape = shape(k, array.shape().d1());
        var pair = operation("topk")
                       .inputs(array)
                       .outputs(prototype(array.type(), shape), prototype(U32, shape))
                       .operation(
                           (leftPtr, rightPtr) -> arrayfire_h.af_topk(leftPtr, rightPtr, array.dereference(), k.size(),
                               0, 0))
                       .grads((results, grads) -> {
                           var values = grads.left();
                           var d0Indices = results.right();
                           var d1Indices = transpose(af.range(values.shape().d1().size())).tileAs(values.shape());
                           return dense(
                               sparse(values.flatten(), d0Indices.cast(S32).flatten(), d1Indices.cast(S32).flatten(),
                                   array.shape()));
                       })
                       .build();
        return new TopKResult<>(pair.left(), pair.right());
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, U, D2, D3>> Array<T, Shape<D0, D0, D2, D3>> diag(
        Array<T, S> array) {
        return operation("diag")
                   .inputs(array)
                   .outputs(prototype(array.type(),
                       shape(array.shape().d0(), array.shape().d0(), array.shape().d2(), array.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_diag_create(ptr, array.dereference(), 0))
                   .grads((result, grads) -> diagExtract(grads).reshape(array.shape()))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D2 extends Num<D2>, D3 extends Num<D3>, SI extends Shape<D0, D0, D2, D3>> Array<T, Shape<D0, U, D2, D3>> diagExtract(
        Array<T, SI> array) {
        return operation("diag_extract")
                   .inputs(array)
                   .outputs(prototype(array.type(),
                       shape(array.shape().d0(), af.u(), array.shape().d2(), array.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_diag_extract(ptr, array.dereference(), 0))
                   .grads((result, grads) -> diag(grads).reshape(array.shape()))
                   .build();
    }

    // https://arrayfire.org/docs/group__blas__func__matmul.htm
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, OD1 extends Num<OD1>, SL extends Shape<D0, D1, D2, D3>, SR extends Shape<D1, OD1, D2, D3>> Array<T, Shape<D0, OD1, D2, D3>> matmul(
        Array<T, SL> left, Array<T, SR> right) {
        if (left.shape().d1().size() != right.shape().d0().size()) {
            throw new IllegalArgumentException(
                String.format("Incompatible shapes for matmul, left: %s right: %s", left.shape(), right.shape()));
        }
        return operation("matmul")
                   .inputs(left, right)
                   .outputs(prototype(left.type(),
                       shape(left.shape().d0(), right.shape().d1(), left.shape().d2(), left.shape().d3())))
                   .operation(ptr -> arrayfire_h.af_matmul(ptr, left.dereference(), right.dereference(), 0, 0))
                   .grads((result, grads) -> {
                       var leftGrads = matmul(grads, transpose(right));
                       var rightGrads = matmul(transpose(left), grads);
                       return new ArrayPair<>(leftGrads.reshape(left.shape()), rightGrads.reshape(right.shape()));
                   })
                   .build();
    }

    public static <T extends DataType<?>, AD0 extends Num<AD0>, AD1 extends Num<AD1>, BD1 extends Num<BD1>, CD1 extends Num<CD1>, D2 extends Num<D2>, D3 extends Num<D3>, SA extends Shape<AD0, AD1, D2, D3>, SB extends Shape<AD1, BD1, D2, D3>, SC extends Shape<BD1, CD1, D2, D3>> Array<T, Shape<AD0, CD1, D2, D3>> matmul(
        Array<T, SA> a, Array<T, SB> b, Array<T, SC> c) {
        if (a.shape().d0().size() * b.shape().d1().size() < b.shape().d0().size() * c.shape().d1().size()) {
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

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> clamp(Array<T, S> array,
                                                                                         Tileable<T, ?> lo,
                                                                                         Tileable<T, ?> hi) {
        assertTileable(array, lo);
        assertTileable(array, hi);
        return clamp(array, lo.array().tileAs(array), hi.array().tileAs(array));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> clamp(Array<T, S> array,
                                                                                         Array<T, S> lo,
                                                                                         Array<T, S> hi) {
        return operation("clamp")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(
                       ptr -> arrayfire_h.af_clamp(ptr, array.dereference(), lo.dereference(), hi.dereference(), true))
                   .grads((result, grads) -> {
                       var loMask = ge(array, lo);
                       var hiMask = le(array, hi);
                       return mul(grads, and(loMask, hiMask).cast(grads.type()));
                   })
                   .build();

    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> relu(Array<T, S> array) {
        return clamp(array, constant(array.type(), 0f).tile(), constant(array.type(), Double.POSITIVE_INFINITY).tile());
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, SL extends S, SR extends S> Array<B8, SL> eq(
        Array<T, SL> left, Array<T, SR> right) {
        return operation("eq")
                   .inputs(left, right)
                   .outputs(prototype(B8, left.shape()))
                   .operation(ptr -> arrayfire_h.af_eq(ptr, left.dereference(), right.dereference(), true))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> negate(Array<T, S> array) {
        var minusOne = constant(array.type(), array.shape(), -1);
        return mul(array, minusOne);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> exp(Array<T, S> array) {
        return operation("exp")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_exp(ptr, array.dereference()))
                   .grads((result, grads) -> mul(grads, result))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> pow(Array<T, S> array, double pow) {
        return pow(array, constant(array.type(), array.shape(), pow));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> pow(Array<T, S> array,
                                                                                       Array<T, S> pow) {
        return operation("pow")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_pow(ptr, array.dereference(), pow.dereference(), false))
                   .grads((result, grads) -> mul(mul(grads, pow),
                       pow(array, sub(pow, constant(pow.type(), pow.shape(), 1)))))
                   .build();
    }

    /**
     * Returns 1 for negative numbers and 0 for positive numbers.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> signbit(Array<T, S> array) {
        return operation("signbit")
                   .inputs(array)
                   .outputs(array.prototype())
                   .operation(ptr -> arrayfire_h.af_sign(ptr, array.dereference()))
                   .grads((result, grads) -> constant(array.type(), array.shape(), 0))
                   .build();
    }

    /**
     * Returns -1 for negative numbers and 1 for positive numbers.
     */
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> signum(Array<T, S> array) {
        return operation("signum")
                   .inputs(array)
                   .outputs(array.prototype())
                   .operation(tidyOperation(() -> sub(af.constant(array.type(), array.shape(), 1),
                       mul(af.constant(array.type(), array.shape(), 2), signbit(array)))))
                   .grads((result, grads) -> constant(array.type(), array.shape(), 0))
                   .build();
    }

    public static Operation.Builder operation(String name) {
        return new Operation.Builder().name(name);
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> log(Array<T, S> array) {
        return operation("log")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_log(ptr, array.dereference()))
                   .grads((result, grads) -> div(grads, array))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> abs(Array<T, S> input) {
        return operation("abs")
                   .inputs(input)
                   .outputs(prototype(input))
                   .operation(ptr -> arrayfire_h.af_abs(ptr, input.dereference()))
                   .grads((result, grads) -> mul(grads, signum(input)))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sqrt(Array<T, S> array) {
        return operation("sqrt")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_sqrt(ptr, array.dereference()))
                   .grads((result, grads) -> div(grads, mul(constant(array.type(), array.shape(), 2), result)))
                   .build();
    }

    public static <ST extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, S> softmax(
        Array<ST, S> array) {
        return softmax(array, 1f);
    }

    public static Function<MemorySegment, Integer> tidyOperation(Supplier<Array<?, ?>> fn) {
        return ptr -> {
            var result = tidy(fn);
            ptr.copyFrom(result.segment());
            Scope.untrack(result);
            return Status.AF_SUCCESS.code();
        };
    }

    public static <ST extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, S> softmax(
        Array<ST, S> array, float temperature) {
        var exp = exp(div(array, constant(array.type(), array.shape(), temperature)));
        return div(exp, sum(exp).tile());
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sigmoid(Array<T, S> array) {
        var one = ones(array);
        return div(one, add(one, exp(negate(array))));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> sparse(Array<T, S> array,
                                                                                          Storage storage) {
        return operation("sparse_from_dense")
                   .inputs(array)
                   .outputs(prototype(array.type(), array.shape()))
                   .operation(
                       ptr -> arrayfire_h.af_create_sparse_array_from_dense(ptr, array.dereference(), storage.code()))
                   .grads((result, grads) -> dense(grads))
                   .build();
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> dense(Array<T, S> array) {
        return operation("dense_from_sparse")
                   .inputs(array)
                   .outputs(prototype(array.type(), array.shape()))
                   .operation(ptr -> arrayfire_h.af_sparse_to_dense(ptr, array.dereference()))
                   .grads((result, grads) -> grads)
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<?, D1, D2, D3>> array, Index<D0> i0) {
        return index(array, i0, seq(array.shape().d1()), seq(array.shape().d2()), seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<?, ?, D2, D3>> array, Index<D0> i0, Index<D1> i1) {
        return index(array, i0, i1, seq(array.shape().d2()), seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<D0, ?, D2, D3>> array, Span ignored0, Index<D1> i1) {
        return index(array, seq(array.shape().d0()), i1, seq(array.shape().d2()), seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<D0, D1, ?, D3>> array, Span ignored0, Span ignored1, Index<D2> i2) {
        return index(array, seq(array.shape().d0()), seq(array.shape().d1()), i2, seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<D0, ?, ?, D3>> array, Span ignored0, Index<D1> i1, Index<D2> i2) {
        return index(array, seq(array.shape().d0()), i1, i2, seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<?, D1, ?, D3>> array, Index<D0> i0, Span ignored1, Index<D2> i2) {
        return index(array, i0, seq(array.shape().d1()), i2, seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<?, ?, ?, D3>> array, Index<D0> i0, Index<D1> i1, Index<D2> i2) {
        return index(array, i0, i1, i2, seq(array.shape().d3()));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ? extends Shape<D0, D1, D2, ?>> array, Span ignored0, Span ignored1, Span ignored2, Index<D3> i3) {
        return index(array, seq(array.shape().d0()), seq(array.shape().d1()), seq(array.shape().d2()), i3);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<T, Shape<D0, D1, D2, D3>> index(
        Array<T, ?> array, Index<D0> i0, Index<D1> i1, Index<D2> i2, Index<D3> i3) {
        return operation("index")
                   .inputs(array)
                   .outputs(
                       prototype(array.type(), shape(i0.createDim(), i1.createDim(), i2.createDim(), i3.createDim())))
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
                       return arrayfire_h.af_index_gen(ptr, array.dereference(), 4, nativeIndexes);
                   })
                   // TODO: Add grads once I work out how to invert and index.
                   .build();

    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>> List<Array<T, Shape<D0, N, U, U>>> batch(
        Array<T, S> array, int batchSize) {
        return batch(array, ArrayFire::n, batchSize);
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>, BDT extends Num<BDT>> List<Array<T, Shape<D0, BDT, U, U>>> batch(
        Array<T, S> array, Function<Integer, BDT> type, int batchSize) {
        var results = new ArrayList<Array<T, Shape<D0, BDT, U, U>>>();
        var d0Seq = seq(array.shape().d0());
        for (int i = 0; i < array.shape().d1().size(); i += batchSize) {
            var computedD1Size = Math.min(batchSize, array.shape().d1().size() - i);
            var slice = index(array, d0Seq, seq(i, i + computedD1Size - 1));
            results.add(slice.reshape(shape(array.shape().d0(), type.apply(computedD1Size))));
        }
        return results;
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>, NS extends Shape<?, ?, ?, ?>> Array<T, NS> tileAs(
        Array<T, S> array, NS newShape) {
        if (newShape.capacity() % array.shape().capacity() != 0) {
            throw new IllegalArgumentException(
                String.format("Can't tile perfectly from %s to %s", array.shape(), newShape));
        }
        int d0ratio = newShape.d0().size() / array.shape().d0().size();
        int d1ratio = newShape.d1().size() / array.shape().d1().size();
        int d2ratio = newShape.d2().size() / array.shape().d2().size();
        int d3ratio = newShape.d3().size() / array.shape().d3().size();
        return operation("tile")
                   .inputs(array)
                   .outputs(prototype(array.type(), newShape))
                   .operation(ptr -> arrayfire_h.af_tile(ptr, array.dereference(), d0ratio, d1ratio, d2ratio, d3ratio))
                   .grads((result, grads) -> sumAs((Array) grads, array.shape()).cast(array.type()))
                   .build();
    }

    @SuppressWarnings({"unchecked", "rawtypes"})
    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, S extends Shape<?, ?, ?, ?>, NS extends Shape<?, ?, ?, ?>> Array<ST, NS> sumAs(
        Array<T, S> input, NS newShape) {
        // I think there is a nicer way to do this in at most two operations.
        Array result = input;
        if (newShape.d0() != input.shape().d0()) {
            if (newShape.d0().size() != 1)
                throw new IllegalArgumentException("Can't sum over D0 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d1() != input.shape().d1()) {
            if (newShape.d1().size() != 1)
                throw new IllegalArgumentException("Can't sum over D1 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d2() != input.shape().d2()) {
            if (newShape.d2().size() != 1)
                throw new IllegalArgumentException("Can't sum over D2 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        if (newShape.d3() != input.shape().d3()) {
            if (newShape.d3().size() != 1)
                throw new IllegalArgumentException("Can't sum over D3 from " + input.shape() + " to " + newShape);
            result = sum(result);
        }
        return reshape(((Array<ST, ?>) result), newShape);
    }

    public static <T extends DataType<?>> Array<T, Shape<N, U, U, U>> flatten(Array<T, ?> array) {
        return reshape(array, shape(array.shape().capacity()));
    }

    public static <T extends DataType<?>, S extends Shape<?, ?, ?, ?>> Array<T, S> flip(Array<T, S> array) {
        return operation("flip")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_flip(ptr, array.dereference(), 0))
                   .grads((result, grads) -> flip(grads))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, FD0 extends Num<FD0>, FD1 extends Num<FD1>, FD3 extends Num<FD3>, S extends Shape<D0, D1, D2, D3>, FS extends Shape<FD0, FD1, D2, FD3>> Array<T, Shape<N, N, FD3, D3>> convolve2(
        Array<T, S> array, Array<T, FS> filters) {
        return convolve2(array, filters, shape(1, 1), shape(0, 0), shape(1, 1));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, FD0 extends Num<FD0>, FD1 extends Num<FD1>, FD3 extends Num<FD3>, S extends Shape<D0, D1, D2, D3>, FS extends Shape<FD0, FD1, D2, FD3>> Array<T, Shape<N, N, FD3, D3>> convolve2(
        Array<T, S> array, Array<T, FS> filters, Shape<?, ?, ?, ?> stride) {
        return convolve2(array, filters, stride, shape(0, 0), shape(1, 1));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, FD0 extends Num<FD0>, FD1 extends Num<FD1>, FD3 extends Num<FD3>, S extends Shape<D0, D1, D2, D3>, FS extends Shape<FD0, FD1, D2, FD3>> Array<T, Shape<N, N, FD3, D3>> convolve2(
        Array<T, S> array, Array<T, FS> filters, Shape<?, ?, ?, ?> stride, Shape<?, ?, ?, ?> padding) {
        return convolve2(array, filters, stride, padding, shape(1, 1));
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, FD0 extends Num<FD0>, FD1 extends Num<FD1>, FD3 extends Num<FD3>, S extends Shape<D0, D1, D2, D3>, FS extends Shape<FD0, FD1, D2, FD3>> Array<T, Shape<N, N, FD3, D3>> convolve2(
        Array<T, S> array, Array<T, FS> filters, Shape<?, ?, ?, ?> stride, Shape<?, ?, ?, ?> padding,
        Shape<?, ?, ?, ?> dilation) {
        if (array.shape().d2().size() != filters.shape().d2().size()) {
            throw new IllegalArgumentException(
                String.format("D2 for input %s and filters %s must match", array.shape(), filters.shape()));
        }
        if (stride.ndims() != 2) {
            throw new IllegalArgumentException(String.format("Stride must be have 2 dims but was %s", stride));
        }
        if (padding.ndims() != 2) {
            throw new IllegalArgumentException(String.format("Padding must be have 2 dims but was %s", padding));
        }
        if (dilation.ndims() != 2) {
            throw new IllegalArgumentException(String.format("Dilation must be have 2 dims but was %s", dilation));
        }
        var computedShape = shape(n((array.shape().d0().size() + 2 * padding.d0().size() -
                                         (filters.shape().d0().size() - 1) * dilation.d0().size() - 1) /
                                        stride.d0().size() + 1),
            n((array.shape().d1().size() + 2 * padding.d1().size() -
                   (filters.shape().d1().size() - 1) * dilation.d1().size() - 1) / stride.d1().size() + 1),
            filters.shape().d3(), array.shape().d3());
        return operation("convolve2")
                   .inputs(array, filters)
                   .outputs(prototype(array.type(), computedShape))
                   .operation(ptr -> {
                       retryWithGc(() -> handleStatus(
                           () -> arrayfire_h.af_convolve2_nn(ptr, array.dereference(), filters.dereference(), 2,
                               nativeDims(stride), 2, nativeDims(padding), 2, nativeDims(dilation))));
                       return Status.AF_SUCCESS.code();
                   })
                   .grads((result, grads) -> {
                       var filterGrads = convolve2(array, grads, stride, padding, dilation);
                   })
                   .build();
    }

    /**
     * L2 norm.
     */
    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, Shape<U, D1, D2, D3>> norm(
        Array<T, S> array) {
        var mul = pow(array, array);
        var sum = sum(mul);
        return sqrt(sum);
    }

    /**
     * Normalize by dividing by the L2 norm.
     */
    public static <ST extends DataType<?>, T extends DataType<? extends DataType.Meta<ST, ?, ?>>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<ST, S> normalize(
        Array<T, S> array) {
        return div(cast(array, array.type().meta().sumType()), norm(array).tile());
    }

    /**
     * Center by subtracting the average.
     */
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>, S extends Shape<D0, D1, D2, D3>> Array<T, S> center(
        Array<T, S> array) {
        return sub(array, mean(array).tile());
    }

    // svd
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>> SvdResult<T, D0, D1> svd(
        Array<T, S> array) {
        var trio = operation("svd")
                       .inputs(array)
                       .outputs(prototype(array.type(), shape(array.shape().d0(), array.shape().d0())),
                           prototype(array.type(), shape(array.shape().d0())),
                           prototype(array.type(), shape(array.shape().d1(), array.shape().d1())))
                       .operation((u, s, v) -> arrayfire_h.af_svd(u, s, v, array.dereference()))
                       .build();
        return new SvdResult<>(trio.left(), trio.middle(), trio.right());
    }

    /**
     * Computes the covariance matrix of the given matrix.
     */
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>> Array<T, Shape<D0, D0, U, U>> cov(
        Array<T, S> array) {
        return tidy(() -> {
            var subMean = sub(array, mean(array, D1).tileAs(array));
            var matrix = matmul(subMean, transpose(subMean));
            return div(matrix, constant(matrix.type(), matrix.shape(), array.shape().d1().size() - 1.0f));
        });
    }

    /**
     * Computes the ZCA whitening matrix of the given matrix.
     */
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>> Array<T, Shape<D0, D0, U, U>> zca(
        Array<T, S> array) {
        return tidy(() -> {
            var cov = cov(array);
            var svd = svd(cov);
            var invSqrtS = diag(div(constant(svd.s().type(), svd.s().shape(), 1.0f),
                sqrt(add(svd.s(), constant(svd.s().type(), svd.s().shape(), 1e-5f)))));
            return matmul(svd.u(), matmul(invSqrtS, transpose(svd.u())));
        });
    }

    /**
     * Inverts the given matrix.
     */
    public static <T extends DataType<?>, D extends Num<D>, S extends Shape<D, D, U, U>> Array<T, S> inverse(
        Array<T, S> array) {
        return operation("inverse")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(ptr -> arrayfire_h.af_inverse(ptr, array.dereference(), 0))
                   .build();
    }

    // TODO: Add uncropped version.
    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, S extends Shape<D0, D1, U, U>> Array<T, S> rotate(
        Array<T, S> array, float angle, InterpolationType interpolationType) {
        return operation("rotate")
                   .inputs(array)
                   .outputs(prototype(array))
                   .operation(
                       ptr -> arrayfire_h.af_rotate(ptr, array.dereference(), angle, true, interpolationType.code()))
                   .grads((result, grads) -> rotate(grads, -angle, interpolationType))
                   .build();
    }

    public static <T extends DataType<?>, D0 extends Num<D0>, D1 extends Num<D1>, ND0 extends Num<ND0>, ND1 extends Num<ND1>, S extends Shape<D0, D1, U, U>> Array<T, Shape<ND0, ND1, U, U>> scale(
        Array<T, S> array, ND0 nd0, ND1 nd1, InterpolationType interpolationType) {
        return operation("scale")
                   .inputs(array)
                   .outputs(prototype(array.type(), shape(nd0, nd1)))
                   .operation(ptr -> arrayfire_h.af_scale(ptr, array.dereference(),
                       (float) nd0.size() / array.shape().d0().size(), (float) nd1.size() / array.shape().d1().size(),
                       nd0.size(), nd1.size(), interpolationType.code()))
                   .grads((result, grads) -> scale(grads, array.shape().d0(), array.shape().d1(),
                       interpolationType).reshape(array.shape()))
                   .build();
    }

    public static <D0 extends Num<D0>, D1 extends Num<D1>, D2 extends Num<D2>, D3 extends Num<D3>> Array<F32, Shape<D0, D1, D2, D3>> oneHot(
        Array<S32, Shape<U, D1, D2, D3>> array, D0 classes) {
        var shape = shape(classes, array.shape().d1(), array.shape().d2(), array.shape().d3());
        return operation("one_hot").inputs(array).outputs(prototype(F32, shape)).operation(tidyOperation(() -> {
            var compressedShape = shape(classes, n(array.shape().capacity()));
            var values = af.constant(F32, array.shape(), 1);
            var d1Indices = af.range(S32, compressedShape.d1().size());
            return dense(sparse(values.flatten(), array.flatten(), d1Indices, compressedShape)).reshape(shape);
        })).build();
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

    public static U u(int ignored) {
        return U;
    }

    public static <T extends Array<?, ?>> T grads(Array<?, ?> loss, T tensor) {
        var graph = new Graph(scope().operations());
        return graph.grads(loss, tensor);
    }

    public static void optimize(Array<?, ?> loss) {
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

    public static DeviceMemInfo deviceMemInfo() {
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
