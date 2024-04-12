using CNN.Layers;

namespace CNN.UnitTests
{
    [TestClass]
    public class SoftMaxLayerTests
    {
        
        [TestMethod]
        public void Forward_Equal_DistributionEqual()
        {
            //Arrange
            var softMaxLayer = new SoftMaxLayer(5);
            Random rnd = new Random();
            int equalInput = rnd.Next(10000); 
            double[] input = {equalInput, equalInput, equalInput, equalInput, equalInput};
            double[] expectedOutput = { 0.2, 0.2, 0.2, 0.2, 0.2 };

            //Act
            double[] output = softMaxLayer.Forward(input);

            //Assert
            CollectionAssert.AreEqual(expectedOutput, output);

        }

        [TestMethod]
        public void Forward_OneOff_OneOff()
        {
            //Arrange
            var softMaxLayer = new SoftMaxLayer(5);
            Random rnd = new Random();
            int intInput = rnd.Next(10000);
            double[] input = { intInput, 0, 0, 0, 0};
            double[] expectedOutput = { 1, 0, 0, 0, 0 };

            //Act
            double[] output = softMaxLayer.Forward(input);

            //Assert
            CollectionAssert.AreEqual(expectedOutput, output);
        }

        [TestMethod]
        public void Backward_Equal_Loss()
        {
            //Arrange
            var softMaxLayer = new SoftMaxLayer(5);
            Random rnd = new Random();
            int equalInput = rnd.Next(10000);
            double[] input = { equalInput, equalInput, equalInput, equalInput, equalInput };
            double[] trueLabel = { 0, 0, 1, 0, 0 };
            double[] expectedLoss = { 0.2, 0.2, -0.8, 0.2, 0.2 };
            //Act
            double[] output = softMaxLayer.Forward(input);
            double[] loss = softMaxLayer.Backward(trueLabel, 1, 1);

            //Assert
            CollectionAssert.AreEqual(expectedLoss, loss);

        }

        [TestMethod]
        public void Backward_OneOff_Loss()
        {
            //Arrange
            var softMaxLayer = new SoftMaxLayer(5);
            Random rnd = new Random();
            int intInput = rnd.Next(10000);
            double[] input = { intInput, 0, 0, 0, 0 };
            double[] trueLabel = { 0, 0, 1, 0, 0 };
            double[] expectedLoss = { 1, 0, -1, 0, 0 };

            //Act
            double[] output = softMaxLayer.Forward(input);
            double[] loss = softMaxLayer.Backward(trueLabel, 1, 1);

            //Assert
            CollectionAssert.AreEqual(expectedLoss, loss);
        }
    }
}