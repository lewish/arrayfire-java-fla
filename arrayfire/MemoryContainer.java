package arrayfire;

import java.lang.foreign.Arena;

/**
 * Indicates this object is backed by a foreign memory segment that must be memory managed.
 */
public interface MemoryContainer {
    Arena arena();
}
