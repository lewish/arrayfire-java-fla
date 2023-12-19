package arrayfire.utils;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

public class IdentityHashSet {
    public static <T> Set<T> create() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }
}
