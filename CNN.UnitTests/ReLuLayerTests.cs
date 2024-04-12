using CNN.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.UnitTests
{
    [TestClass]
    public class ReLuLayerTests
    {
        [TestMethod]
        public void Forward_OneOff_OneOff()
        {
            //Arrange
            var reluLayer = new ReLuLayer(5);
            double[] input = { 1, -1, -1, -1, -1 };
            double[] expectedOutput = { 1, 0, 0, 0, 0 };

            //Act
            double[] output = reluLayer.Forward(input);

            //Assert
            CollectionAssert.AreEqual(expectedOutput, output);
        }

        [TestMethod]
        public void Forward_Rand_RandPositive()
        {
            //Arrange
            var reluLayer = new ReLuLayer(5);
            Random rnd = new Random();
            double[] input = new double[5];
            double[] expectedOutput = new double[5];

            //Act
            for (int i = 0; i < 5; i++)
            {
                input[i] = rnd.NextDouble();
            }

            for (int i = 0; i < 5; i++)
            {
                if (input[i] > 0)
                {
                    expectedOutput[i] = input[i];
                }
                else
                {
                    expectedOutput[i] = 0;
                }

            }

            double[] output = reluLayer.Forward(input);

            //Assert
            CollectionAssert.AreEqual(expectedOutput, output);
        }

        [TestMethod]
        public void Backward_OneOff_OneOff()
        {
            //Arrange
            var reluLayer = new ReLuLayer(5);
            double[] input = { 1, -1, -1, -1, -1 };
            double[] inputLoss = { -10, 5, 6, -8, 7 };
            double[] expectedLoss = { -10, 0, 0, 0, 0 };
            //Act
            double[] output = reluLayer.Forward(input);
            double[] loss = reluLayer.Backward(inputLoss, 1, 1);

            //Assert
            CollectionAssert.AreEqual(expectedLoss, loss);
        }

        [TestMethod]
        public void Backward_Rand_RandPositive()
        {
            //Arrange
            var reluLayer = new ReLuLayer(5);
            Random rnd = new Random();
            double[] input = new double[5];
            double[] inputLoss = { -10, 5, 6, -8, 7 };
            double[] expectedLoss = new double[5];

            //Act
            for (int i = 0; i < 5; i++)
            {
                input[i] = rnd.NextDouble();
            }

            for (int i = 0; i < 5; i++)
            {
                if (input[i] > 0)
                {
                    expectedLoss[i] = inputLoss[i];
                }
                else
                {
                    expectedLoss[i] = 0;
                }

            }

            double[] output = reluLayer.Forward(input);
            double[] loss = reluLayer.Backward(inputLoss, 1, 1);

            //Assert
            CollectionAssert.AreEqual(expectedLoss, loss);
        }
    }
}
