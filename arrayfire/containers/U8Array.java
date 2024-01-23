package arrayfire.containers;

import arrayfire.U8;

import java.lang.foreign.ValueLayout;

import static arrayfire.ArrayFire.U8;

public class U8Array extends NativeArray<U8, Byte, byte[]> {

    public U8Array(int length) {
        super(U8, length);
    }

    @Override
    public ValueLayout.OfByte layout() {
        return ValueLayout.JAVA_BYTE;
    }

    @Override
    public Byte get(int index) {
        return segment.getAtIndex(layout(), index);
    }

    @Override
    public void set(int index, Byte value) {
        segment.setAtIndex(layout(), index, value);
    }

    @Override
    public byte[] java() {
        var array = new byte[length];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
}
