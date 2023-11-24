package fade.flags;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FlagsTest {

  public enum TestEnum {
    A, B;
  }

  @Test
  public void enums() {
    Flags.parseForTesting("entrypoint", "--enum", "B");
    var enumFlag = Flags.enumFlag("enum", TestEnum.values());
    Assert.assertEquals(TestEnum.B, enumFlag.get());
  }

  @Test
  public void doubles() {
    Flags.parseForTesting("entrypoint", "--double", "0.4234");
    var enumFlag = Flags.doubleFlag("double");
    Assert.assertEquals(0.4234, (double) enumFlag.get(), 0.0);
  }
}
