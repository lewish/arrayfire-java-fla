package arrayfire;

import arrayfire.datatypes.AfDataType;

import java.lang.foreign.MemorySegment;

import static arrayfire.ArrayFire.af;

public record HostTensor<T extends AfDataType<?>, D0 extends Number, D1 extends Number, D2 extends Number, D3 extends Number>(
    MemorySegment segment, T type, Shape<D0, D1, D2, D3> shape) {


  public int capacity() {
    return shape.capacity();
  }

  public <JT> JT get(AfDataType<JT> dt, int index) {
    return dt.get(segment, index);
  }

  public <JT> void set(AfDataType<JT> dt, int index, JT value) {
    dt.set(segment, index, value);
  }

  public Tensor<T, D0, D1, D2, D3> push() {
    return af.push(this);
  }

  @Override
  public String toString() {
    StringBuilder str = new StringBuilder("[");
    var max = 1000;
    var lim = Math.min(capacity(), max);
    for (int i = 0; i < lim; i++) {
      str.append(get((AfDataType<?>) type, i));
      if (i < lim - 1) {
        str.append(", ");
      }
    }
    if (capacity() > max) {
      str.append(", ...");
    }
    str.append("]");
    return str.toString();
  }
}
