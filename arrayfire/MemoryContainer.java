package arrayfire;

/**
 * Indicates this object contains foreign memory that must be manually managed.
 */
public interface MemoryContainer {

    /**
     * Free the memory associated with this object.
     */
    void dispose();
}
