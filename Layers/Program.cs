using CNN.Layers;

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
        int numFilters = 2; // Number of filters (output channels)

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

        double[,,,] output = convLayer.Forward(input);
        convLayer.Display();

        // Example target output (just for demonstration)
        double[,,,] targetOutput = new double[output.GetLength(0), output.GetLength(1), output.GetLength(2), output.GetLength(3)]; // Your target output here

        // Compute loss (just for demonstration)
        double[,,,] loss = new double[output.GetLength(0), output.GetLength(1), output.GetLength(2), output.GetLength(3)];
        for (int f = 0; f < output.GetLength(3); f++)
        {
            for (int d = 0; d < output.GetLength(2); d++)
            {
                for (int i = 0; i < output.GetLength(0); i++)
                {
                    for (int j = 0; j < output.GetLength(1); j++)
                    {
                        loss[i, j, d, f] = output[i, j, d, f] - targetOutput[i, j, d, f]; // Simple difference for demonstration
                    }
                }
            }
        }

        // Backpropagation
        convLayer.Backward(loss);

        // Update weights (with a chosen learning rate)
        double learningRate = 0.01;
        convLayer.UpdateWeights(learningRate);
        convLayer.Display();
    }
}
