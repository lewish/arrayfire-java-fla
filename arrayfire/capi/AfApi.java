package arrayfire.capi;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.List;

public class AfApi {

  // See https://github.com/arrayfire/arrayfire/tree/master/include/af
  // See https://openjdk.org/jeps/419

  private static AfApi instance;

  public final AfNativeMethod1<MemorySegment> getAvailableBackends = new AfNativeMethod1<>("af_get_available_backends",
      MemorySegment.class);
  public final AfNativeMethod1<MemorySegment> getActiveBackend = new AfNativeMethod1<>("af_get_active_backend",
      MemorySegment.class);
  public final AfNativeMethod1<Integer> setBackend = new AfNativeMethod1<>("af_set_backend", int.class);
  public final AfNativeMethod1<MemorySegment> getDevice = new AfNativeMethod1<>("af_get_device", MemorySegment.class);
  public final AfNativeMethod1<MemorySegment> getDeviceCount = new AfNativeMethod1<>("af_get_device_count",
      MemorySegment.class);
  public final AfNativeMethod1<Integer> setDevice = new AfNativeMethod1<>("af_set_device", int.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, MemorySegment> deviceInfo = new AfNativeMethod4<>("af_device_info",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, MemorySegment> getVersion = new AfNativeMethod3<>("af_get_version",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class);

  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, MemorySegment> deviceMemInfo = new AfNativeMethod4<>("af_device_mem_info",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class);

  public final AfNativeMethod2<MemorySegment, Integer> printMemInfo = new AfNativeMethod2<>("af_print_mem_info",
      MemorySegment.class,
      int.class);

  public final AfNativeMethod0 deviceGc = new AfNativeMethod0("af_device_gc");
  public final MethodHandle getLastError = Linker.nativeLinker().downcallHandle(
      SymbolLookup.loaderLookup().find("af_get_last_error").orElseThrow(ExceptionInInitializerError::new),
      FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

  public final AfNativeMethod5<MemorySegment, MemorySegment, Integer, MemorySegment, Integer> createArray = new AfNativeMethod5<>("af_create_array",
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod5<MemorySegment, Double, Integer, MemorySegment, Integer> constant = new AfNativeMethod5<>("af_constant",
      MemorySegment.class,
      double.class,
      int.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod2<MemorySegment, MemorySegment> getArrayData = new AfNativeMethod2<>("af_get_data_ptr",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod1<MemorySegment> releaseArray = new AfNativeMethod1<>("af_release_array",
      MemorySegment.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> retainArray = new AfNativeMethod2<>("af_retain_array",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod1<MemorySegment> eval = new AfNativeMethod1<>("af_eval", MemorySegment.class);
  public final AfNativeMethod1<Integer> sync = new AfNativeMethod1<>("af_sync", int.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> getDataRefCount = new AfNativeMethod2<>("af_get_data_ref_count",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod1<MemorySegment> freeDeviceV2 = new AfNativeMethod1<>("af_free_device_v2",
      MemorySegment.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Boolean> transpose = new AfNativeMethod3<>("af_transpose",
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, Integer, MemorySegment> moddims = new AfNativeMethod4<>("af_moddims",
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> mul = new AfNativeMethod4<>("af_mul",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> div = new AfNativeMethod4<>("af_div",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> add = new AfNativeMethod4<>("af_add",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> sub = new AfNativeMethod4<>("af_sub",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);

  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> ge = new AfNativeMethod4<>("af_ge",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);

  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> maxof = new AfNativeMethod4<>("af_maxof",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);

  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> minof = new AfNativeMethod4<>("af_minof",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod5<MemorySegment, MemorySegment, MemorySegment, MemorySegment, Boolean> clamp = new AfNativeMethod5<>("af_clamp",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);

  public final AfNativeMethod4<MemorySegment, Integer, MemorySegment, MemorySegment> join = new AfNativeMethod4<>("af_join",
      MemorySegment.class,
      int.class,
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod5<MemorySegment, MemorySegment, MemorySegment, Integer, Integer> matmul = new AfNativeMethod5<>("af_matmul",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      int.class);
  public final AfNativeMethod5<MemorySegment, MemorySegment, MemorySegment, Integer, Integer> dot = new AfNativeMethod5<>("af_dot",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      int.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> sum = new AfNativeMethod3<>("af_sum",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> mean = new AfNativeMethod3<>("af_mean",
      MemorySegment.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> median = new AfNativeMethod3<>("af_median",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> max = new AfNativeMethod3<>("af_max",
      MemorySegment.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> min = new AfNativeMethod3<>("af_min",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Integer> imax = new AfNativeMethod4<>("af_imax",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod6<MemorySegment, MemorySegment, MemorySegment, Integer, Integer, Integer> topk = new AfNativeMethod6<>("af_topk",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      int.class,
      int.class);


  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, MemorySegment> svd = new AfNativeMethod4<>("af_svd",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class);

  public final AfNativeMethod4<MemorySegment, MemorySegment, MemorySegment, Boolean> eq = new AfNativeMethod4<>("af_eq",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      boolean.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> cast = new AfNativeMethod3<>("af_cast",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> flip = new AfNativeMethod3<>("af_flip",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> exp = new AfNativeMethod2<>("af_exp",
      MemorySegment.class,
      MemorySegment.class);

  public final AfNativeMethod2<MemorySegment, MemorySegment> log = new AfNativeMethod2<>("af_log",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> abs = new AfNativeMethod2<>("af_abs",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> sqrt = new AfNativeMethod2<>("af_sqrt",
      MemorySegment.class,
      MemorySegment.class);

  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> inverse = new AfNativeMethod3<>("af_inverse",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod6<MemorySegment, MemorySegment, Integer, Integer, Integer, Integer> tile = new AfNativeMethod6<>("af_tile",
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      int.class,
      int.class,
      int.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, Integer, MemorySegment> index = new AfNativeMethod4<>("af_index",
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class);
  public final AfNativeMethod4<MemorySegment, MemorySegment, Integer, MemorySegment> indexGen = new AfNativeMethod4<>("af_index_gen",
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class);

  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> diagCreate = new AfNativeMethod3<>("af_diag_create",
      MemorySegment.class,
      MemorySegment.class,
      int.class);
  public final AfNativeMethod9<MemorySegment, MemorySegment, MemorySegment, Integer, MemorySegment, Integer, MemorySegment, Integer, MemorySegment> convolve2Nn = new AfNativeMethod9<>(
      "af_convolve2_nn",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class,
      int.class,
      MemorySegment.class);
  public final AfNativeMethod2<MemorySegment, MemorySegment> getType = new AfNativeMethod2<>("af_get_type",
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod5<MemorySegment, MemorySegment, MemorySegment, MemorySegment, MemorySegment> getDims = new AfNativeMethod5<>(
      "af_get_dims",
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class,
      MemorySegment.class);
  public final AfNativeMethod3<MemorySegment, MemorySegment, Integer> sparseFromDense = new AfNativeMethod3<>("af_create_sparse_array_from_dense",
      MemorySegment.class,
      MemorySegment.class,
      int.class);

  public final AfNativeMethod5<MemorySegment, MemorySegment, Float, Boolean, Integer> rotate = new AfNativeMethod5<>("af_rotate",
      MemorySegment.class,
      MemorySegment.class,
      float.class,
      boolean.class,
      int.class);

  public final AfNativeMethod7<MemorySegment, MemorySegment, Float, Float, Long, Long, Integer> scale = new AfNativeMethod7<>("af_scale",
      MemorySegment.class,
      MemorySegment.class,
      float.class,
      float.class,
      long.class,
      long.class,
      int.class);

  public final AfNativeMethod5<MemorySegment, MemorySegment, Float, Boolean, Integer> translate = new AfNativeMethod5<>("af_translate",
      MemorySegment.class,
      MemorySegment.class,
      float.class,
      boolean.class,
      int.class);

  public final AfNativeMethod5<MemorySegment, MemorySegment, Float, Boolean, Integer> skew = new AfNativeMethod5<>("af_skew",
      MemorySegment.class,
      MemorySegment.class,
      float.class,
      boolean.class,
      int.class);


  public static AfApi get() {
    if (instance == null) {
      loadNativeLibraries();
      var api = new AfApi();
      try (var scope = Arena.ofConfined()) {
        var result = scope.allocateArray(ValueLayout.JAVA_INT, 3);
        api.getVersion.invoke(result, result.asSlice(4), result.asSlice(8));
        var version = result.toArray(ValueLayout.JAVA_INT);
        if (version[0] != 3 || version[1] < 8) {
          throw new IllegalStateException(String.format("Unsupported ArrayFire version, must be >= 3.8.0: %s",
              Arrays.toString(version)));
        }
      }
      instance = api;
    }
    return instance;
  }

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

}