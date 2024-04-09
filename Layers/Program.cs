using CNN.Layers;
using System;

class Program
{
    static void Main(string[] args)
    {
        // Example usage
        int inputDepth = 3;
        int inputWidth = 5;
        int inputHeight = 5;
        int kernelSize = 3;
        int stride = 1;
        int numFilters = 3; // Number of filters (output channels)

        ConvolutionLayer convLayer = new ConvolutionLayer(inputDepth, inputWidth, inputHeight, kernelSize, stride, numFilters);

        // Example input
        double[,,] input = new double[,,]
        {
            {
                {1, 2, 3, 4, 5},
                {6, 7, 8, 9, 10},
                {11, 12, 13, 14, 15},
                {16, 17, 18, 19, 20},
                {21, 22, 23, 24, 25}
            },
            {
                {26, 27, 28, 29, 30},
                {31, 32, 33, 34, 35},
                {36, 37, 38, 39, 40},
                {41, 42, 43, 44, 45},
                {46, 47, 48, 49, 50}
            },
            {
                {51, 52, 53, 54, 55},
                {56, 57, 58, 59, 60},
                {61, 62, 63, 64, 65},
                {66, 67, 68, 69, 70},
                {71, 72, 73, 74, 75}
            }
        };

        double[,,] output = convLayer.Forward(input);
        

        // Example target output (just for demonstration)
        double[,,] targetOutput = new double[output.GetLength(0), output.GetLength(1), output.GetLength(2)]; // Your target output here
        Random random = new Random();
        for (int d = 0; d < output.GetLength(0); d++)
        {
            for (int i = 0; i < output.GetLength(1); i++)
            {
                for (int j = 0; j < output.GetLength(2); j++)
                {
                    targetOutput[d, i, j] = random.NextDouble(); // Simple difference for demonstration
                }
            }
        }
        double[,,] loss = new double[output.GetLength(0), output.GetLength(1), output.GetLength(2)];

        for (int e = 0; e < 10; e++)
        {


            // Compute loss (just for demonstration)

            for (int d = 0; d < output.GetLength(0); d++)
            {
                for (int i = 0; i < output.GetLength(1); i++)
                {
                    for (int j = 0; j < output.GetLength(2); j++)
                    {
                        loss[d, i, j] = output[d, i, j] - targetOutput[d, i, j]; // Simple difference for demonstration
                    }
                }
            }


            // Update weights (with a chosen learning rate)
            double learningRate = 0.0001;
            double momentum = 0;
            // Backpropagation
            double[,,] temp = convLayer.Backward(loss, learningRate, momentum);
            output = convLayer.Forward(input);
            
            for (int d = 0; d < output.GetLength(0); d++)
            {
                for (int i = 0; i < output.GetLength(1); i++)
                {
                    for (int j = 0; j < output.GetLength(2); j++)
                    {
                        Console.Write(loss[d, i, j]);
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
                Console.WriteLine();
            }

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
            
            /*
            for (int d = 0; d < convLayer.weights.GetLength(0); d++)
            {
                for (int f = 0; f < convLayer.weights.GetLength(1); f++)
                {
                    for (int i = 0; i < convLayer.weights.GetLength(2); i++)
                    {
                        for (int j = 0; j < convLayer.weights.GetLength(3); j++)
                        {
                            Console.Write(convLayer.weights[d, f, i, j]);
                        }

                    }

                    Console.WriteLine();
                }
                Console.WriteLine();
                Console.WriteLine();
            }
            */
        }

    }
}
