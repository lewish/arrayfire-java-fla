package arrayfire.capi;

import arrayfire.AfStatus;
import arrayfire.ArrayFireException;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.Map;

import static arrayfire.ArrayFire.af;

public class AfNativeMethod {
  public static final Map<Class<?>, MemoryLayout> ARG_TYPE_LAYOUT_MAP = Map.of(
      MemorySegment.class,
      ValueLayout.ADDRESS,
      boolean.class,
      ValueLayout.JAVA_BOOLEAN,
      byte.class,
      ValueLayout.JAVA_BYTE,
      short.class,
      ValueLayout.JAVA_SHORT,
      int.class,
      ValueLayout.JAVA_INT,
      long.class,
      ValueLayout.JAVA_LONG,
      float.class,
      ValueLayout.JAVA_FLOAT,
      double.class,
      ValueLayout.JAVA_DOUBLE);

  protected final MethodHandle handle;

  AfNativeMethod(String name, Class<?>... classes) {
    var argLayouts = Arrays.stream(classes).map(ARG_TYPE_LAYOUT_MAP::get).toArray(MemoryLayout[]::new);
    handle = Linker.nativeLinker().downcallHandle(
        SymbolLookup.loaderLookup().find(name).orElseThrow(ExceptionInInitializerError::new),
        FunctionDescriptor.of(ValueLayout.JAVA_INT, argLayouts));

  }

  public static void handleArrayFireResponse(Object res) {
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
