package arrayfire;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public record Seq(double begin, double end, double step) {

  public static MemoryLayout LAYOUT = MemoryLayout.structLayout(
      ValueLayout.JAVA_DOUBLE.withName("begin"),
      ValueLayout.JAVA_DOUBLE.withName("end"),
      ValueLayout.JAVA_DOUBLE.withName("step"));

  public void emigrate(MemorySegment segment) {
    segment.set(ValueLayout.JAVA_DOUBLE, LAYOUT.byteOffset(PathElement.groupElement("begin")), begin());
    segment.set(ValueLayout.JAVA_DOUBLE, LAYOUT.byteOffset(PathElement.groupElement("end")), end());
    segment.set(ValueLayout.JAVA_DOUBLE, LAYOUT.byteOffset(PathElement.groupElement("step")), step());
  }

  public int size() {
    return ((int) Math.max(end, begin) - ((int) Math.min(end, begin)) + 1) / (int) Math.abs(step);
  }
}
