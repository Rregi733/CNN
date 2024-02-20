using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class ConvolutionLayer : ILayer
    {
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private int kernelSize;
        private int stride;
        private int numFilters; // Number of filters (output channels)
        private double[,,,] weights; // Weights for each filter
        private double[] biases; // Biases for each filter

        private double[,,] input;
        private double[,,,] gradientWeights;
        private double[] gradientBiases;

        public ConvolutionLayer(int inputDepth, int inputWidth, int inputHeight, int kernelSize, int stride, int numFilters)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.kernelSize = kernelSize;
            this.stride = stride;
            this.numFilters = numFilters;
            input = new double[inputDepth, inputWidth, inputHeight];

            // Initialize weights and biases randomly or with specific values
            Random random = new Random();
            weights = new double[kernelSize, kernelSize, inputDepth, numFilters];
            gradientWeights = new double[kernelSize, kernelSize, inputDepth, numFilters];
            biases = new double[numFilters];
            gradientBiases = new double[numFilters];
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        for (int k = 0; k < inputDepth; k++)
                        {
                            weights[i, j, k, f] = random.NextDouble() * 0.1; // Initialize with small random values
                        }
                    }
                }
                biases[f] = random.NextDouble() * 0.1; // Initialize bias with a small random value
            }
        }

        public double[,,,] Forward(double[,,] input)
        {
            this.input = input;
            int outputDepth = numFilters;
            int outputWidth = (inputWidth - kernelSize) / stride + 1;
            int outputHeight = (inputHeight - kernelSize) / stride + 1;
            double[,,,] output = new double[outputWidth, outputHeight, outputDepth, numFilters];

            for (int f = 0; f < numFilters; f++)
            {
                for (int d = 0; d < outputDepth; d++)
                {
                    for (int i = 0; i < outputWidth; i++)
                    {
                        for (int j = 0; j < outputHeight; j++)
                        {
                            // Apply convolution operation
                            double sum = 0;
                            for (int m = 0; m < kernelSize; m++)
                            {
                                for (int n = 0; n < kernelSize; n++)
                                {
                                    for (int k = 0; k < inputDepth; k++)
                                    {
                                        int inputX = i * stride + m;
                                        int inputY = j * stride + n;
                                        sum += input[k, inputX, inputY] * weights[m, n, k, f];
                                    }
                                }
                            }
                            output[i, j, d, f] = sum + biases[f];
                        }
                    }
                }
            }

            return output;
        }

        public void Backward(double[,,,] dLoss_dY, double learningRate)
        {
            int outputDepth = numFilters;
            int outputWidth = (inputWidth - kernelSize) / stride + 1;
            int outputHeight = (inputHeight - kernelSize) / stride + 1;

            for (int f = 0; f < numFilters; f++)
            {
                for (int d = 0; d < outputDepth; d++)
                {
                    for (int i = 0; i < outputWidth; i++)
                    {
                        for (int j = 0; j < outputHeight; j++)
                        {
                            for (int m = 0; m < kernelSize; m++)
                            {
                                for (int n = 0; n < kernelSize; n++)
                                {
                                    for (int k = 0; k < inputDepth; k++)
                                    {
                                        int inputX = i * stride + m;
                                        int inputY = j * stride + n;
                                        gradientWeights[m, n, k, f] += input[k, inputX, inputY] * dLoss_dY[i, j, d, f];
                                    }
                                }
                            }
                            gradientBiases[f] += dLoss_dY[i, j, d, f];
                        }
                    }
                }
            }
            this.UpdateWeights(learningRate);
        }

        public void UpdateWeights(double learningRate)
        {
            // Update weights and biases using the computed gradients
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        for (int k = 0; k < inputDepth; k++)
                        {
                            weights[i, j, k, f] -= learningRate * gradientWeights[i, j, k, f];
                            gradientWeights[i, j, k, f] = 0; // Reset gradient
                        }
                    }
                }
                biases[f] -= learningRate * gradientBiases[f];
                gradientBiases[f] = 0; // Reset gradient
            }
        }

        public void Display()
        {
            foreach (double x in weights)
            {
                System.Console.WriteLine(x);
            }
        }
    }
}
