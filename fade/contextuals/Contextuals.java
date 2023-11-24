package fade.contextuals;

import fade.context.Context;
import fade.context.Contextual;

import java.util.Random;

import static fade.common.Memoizer.memoize;

public class Contextuals {
  public static class Seed {
    private final long seed;
    private final Random random;

    public static Seed create(long seed) {
      return new Seed(seed);
    }

    private Seed(long seed) {
      this.seed = seed;
      this.random = new Random(seed);
    }

    public long seed() {
      return seed;
    }

    public Random random() {
      return random;
    }
  }


  public static final Contextual<Seed> SEED = Contextual.named("seed", memoize(() -> Seed.create(0)));

  public static Seed seed() {
    return SEED.get();
  }

  public static Context.Entry<Seed> seed(long seed) {
    return new Context.Entry<>(SEED, Seed.create(seed));
  }

  public static final Contextual<Integer> EPOCH = Contextual.named("epoch");

  public static Integer epoch() {
    return EPOCH.get();
  }

  public static Context.Entry<Integer> epoch(int epoch) {
    return new Context.Entry<>(EPOCH, epoch);
  }

  public static final Contextual<Boolean> EAGER = Contextual.named("eager");

  public static Boolean eager() {
    return EAGER.get();
  }

  public static Context.Entry<Boolean> eager(boolean eager) {
    return new Context.Entry<>(EAGER, eager);
  }
}
