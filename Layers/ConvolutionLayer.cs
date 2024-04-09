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
        //parameters of the input
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private double[,,] input;

        //parameters of the filters
        private int kernelSize; 
        private int stride;    
        private int numFilters; 
        public double[,,,] weights; 
        private double[] biases; 

        //gradients wrt filters and biases
        public double[,,,] gradientWeights;
        private double[] gradientBiases;

        //gradients wrt input
        private double[,,,] velocityWeights;
        private double[] velocityBiases;
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
            weights = new double[numFilters, inputDepth, kernelSize, kernelSize];
            biases = new double[numFilters];

            gradientWeights = new double[numFilters, inputDepth, kernelSize, kernelSize];
            gradientBiases = new double[numFilters];

            velocityWeights = new double[numFilters, inputDepth, kernelSize, kernelSize];
            velocityBiases = new double[numFilters];

            for (int f = 0; f < numFilters; f++)
            {
                for (int k = 0; k < inputDepth; k++)
                {
                    for (int i = 0; i < kernelSize; i++)
                    {
                        for (int j = 0; j < kernelSize; j++)
                        {
                            weights[f, k, i, j] = random.NextDouble() * 0.1; // Initialize with small random values
                        }
                    }
                }
                biases[f] = random.NextDouble() * 0.1; // Initialize bias with a small random value
            }

            for (int f = 0; f < numFilters; f++)
            {
                for (int k = 0; k < inputDepth; k++)
                {
                    for (int i = 0; i < kernelSize; i++)
                    {
                        for (int j = 0; j < kernelSize; j++)
                        {
                            velocityWeights[f, k, i, j] = 0; // Initialize with 0
                        }
                    }
                }
                velocityBiases[f] = 0; // Initialize velocityBiases with 0
            }
        }

        // Convolution operation in feed forward
        public double[,,] Forward(double[,,] input)
        {
            this.input = input;
            int outputWidth = (inputWidth - kernelSize) / stride + 1;
            int outputHeight = (inputHeight - kernelSize) / stride + 1;
            double[,,] output = new double[numFilters, outputWidth, outputHeight];

            for (int f = 0; f < numFilters; f++)
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
                                        sum += input[k, inputX, inputY] * weights[f, k, m, n];
                                    }
                                }
                            }
                            output[f, i, j] = sum + biases[f];
                        }
                    }
                
            }

            return output;
        }

        // Backpropagation
        public double[,,] Backward(double[,,] dLoss_dY, double learningRate, double momentum)
        {
            int outputWidth = (inputWidth - kernelSize) / stride + 1;
            int outputHeight = (inputHeight - kernelSize) / stride + 1;

            double[,,] inputGradient = new double[inputDepth, inputWidth, inputHeight];

            //Calculating  gradient of loss wrt input
            for (int d = 0; d < inputDepth; d++)
            {
                for (int iw = 0; iw < inputWidth; iw++)
                {
                    for (int ih = 0; ih < inputHeight; ih++)
                    {
                        double inputGradSum = 0;
                        for (int fd = 0; fd < numFilters; fd++)
                        {
                            for (int fw = 0; fw < kernelSize; fw++)
                            {
                                for (int fh = 0; fh < kernelSize; fh++)
                                {
                                    // Calculate output position corresponding to this input position and filter position
                                    int outputX = iw - kernelSize + fw + 1;
                                    int outputY = ih - kernelSize + fh + 1;

                                    // Check if input position is within bounds
                                    if (outputX >= 0 && outputX < outputWidth && outputY >= 0 && outputY < outputHeight)
                                    {
                                        // Update input gradient using cross-correlation with rotated filter weights
                                        inputGradSum += dLoss_dY[fd, outputX, outputY] * weights[fd, d, kernelSize - fw - 1, kernelSize - fh - 1];
                                    }
                                }
                            }
                        }
                        inputGradient[d,iw,ih]=inputGradSum;
                    }
                }
            }

            // Gradient with respect to weights and biases
            for (int fd = 0; fd < numFilters; fd++)
            {
                for (int d = 0; d < inputDepth; d++)
                {
                    for (int fw = 0; fw < kernelSize; fw++)
                    {
                        for (int fh = 0; fh < kernelSize; fh++)
                        {
                            double weightGradSum = 0;
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                for (int oh = 0; oh < outputHeight; oh++)
                                {
                                    int inputX = ow * stride + fw;
                                    int inputY = oh * stride + fh;
                                    if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight)
                                    {
                                        weightGradSum += input[d, inputX, inputY] * dLoss_dY[fd, ow, oh];
                                    }
                                }
                            }
                            gradientWeights[fd, d, fw, fh] = weightGradSum;
                        }
                    }
                }

                double biasGradSum = 0;
                for (int ow = 0; ow < outputWidth; ow++)
                {
                    for (int oh = 0; oh < outputHeight; oh++)
                    {
                        biasGradSum += dLoss_dY[fd, ow, oh];
                    }
                }
                gradientBiases[fd] = biasGradSum;
            }

            this.UpdateWeights(learningRate, momentum);

            return inputGradient;
        }

        public void UpdateWeights(double learningRate, double momentum)
        {
            // Update weights and biases using SGD with momentum
            for (int f = 0; f < numFilters; f++)
            {
                for (int i = 0; i < kernelSize; i++)
                {
                    for (int j = 0; j < kernelSize; j++)
                    {
                        for (int k = 0; k < inputDepth; k++)
                        {
                            // Compute velocity update
                            velocityWeights[f, k, i, j] = momentum * velocityWeights[f, k, i, j] + learningRate * gradientWeights[f, k, i, j];
                            // Update weights using momentum
                            weights[f, k, i, j] -= velocityWeights[f, k, i, j];
                            // Reset gradient
                            gradientWeights[f, k, i, j] = 0;
                        }
                    }
                }
                // Compute velocity update for biases
                velocityBiases[f] = momentum * velocityBiases[f] + learningRate * gradientBiases[f];
                // Update biases using momentum
                biases[f] -= velocityBiases[f];
                // Reset gradient
                gradientBiases[f] = 0;
            }
        }
    }
}
