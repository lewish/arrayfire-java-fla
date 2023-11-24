package arrayfire;

import fade.context.Contextual;

import java.util.HashSet;
import java.util.Set;

public class Scope implements AutoCloseable {

  public static final Contextual<Scope> CONTEXTUAL = Contextual.named("arrayfire_memory_scope");

  public final Scope parent;
  private final Set<Tensor<?, ?, ?, ?, ?>> tensors = new HashSet<>();
  private final Set<HostTensor<?, ?, ?, ?, ?>> hostTensors = new HashSet<>();

  public Scope(Scope parent) {
    this.parent = parent;
  }

  @Override
  public void close() {
    tensors.forEach(ArrayFire::release);
    hostTensors.forEach(ArrayFire::release);
  }

  public void track(Tensor<?, ?, ?, ?, ?> tensor) {
    this.tensors.add(tensor);
  }

  public void track(HostTensor<?, ?, ?, ?, ?> hostTensor) {
    this.hostTensors.add(hostTensor);
  }

  public void untrack(Tensor<?, ?, ?, ?, ?> tensor) {
    this.tensors.remove(tensor);
  }

  public void untrack(HostTensor<?, ?, ?, ?, ?> hostTensor) {
    this.hostTensors.remove(hostTensor);
  }
}
