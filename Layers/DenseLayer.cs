using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class DenseLayer : ILayer
    {
        private int inputSize;
        private int outputSize;
        private double[,] weights;
        private double[] biases;
        private double[] inputs;
        private double[] outputs;

        public double[,] gradientWeights;
        private double[] gradientBiases;

        //gradients wrt input
        private double[,] velocityWeights;
        private double[] velocityBiases;
        public DenseLayer(int inputSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            inputs = new double[inputSize];
            outputs = new double[outputSize];

            gradientWeights = new double[inputSize, outputSize];
            gradientBiases = new double[outputSize];

            velocityWeights = new double[inputSize, outputSize];
            velocityBiases = new double[outputSize];

            // Initialize weights and biases randomly
            Random rand = new Random();
            weights = new double[inputSize, outputSize];
            biases = new double[outputSize];
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weights[i, j] = rand.NextDouble() * 0.1 - 0.05; // Random weights between -0.05 and 0.05
                }
            }

            for (int i = 0; i < outputSize; i++)
            {
                biases[i] = rand.NextDouble() * 0.1 - 0.05; // Random biases between -0.05 and 0.05
            }

            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    
                    velocityWeights[j, i] = 0;

                }

                gradientBiases[i] = 0;
            }
        }

        public double[] Forward(double[] inputs)
        {
            this.inputs = inputs;

            outputs = new double[outputSize];

            // Perform matrix multiplication
            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputSize; j++)
                {
                    sum += inputs[j] * weights[j, i];
                }
                outputs[i] = sum + biases[i];
            }

            return outputs;
        }

        public double[] Backward(double[] dLoss_dY, double learningRate, double momentum)
        {
            double[] inputGradient = new double[inputSize];

            //Calculating  gradient of loss wrt input
            for (int i = 0; i < inputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < outputSize; j++)
                {
                    sum += dLoss_dY[j] * weights[i, j];
                }
                inputGradient[i] = sum;
            }

            // Gradient with respect to weights 

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    gradientWeights[i, j] = inputs[i] * dLoss_dY[j];
                }

            }

            // Gradient with respect to biases 
            gradientBiases = dLoss_dY;

            this.UpdateWeights(learningRate, momentum);

            return inputGradient;
        }

        public void UpdateWeights(double learningRate, double momentum)
        {
            // Update weights and biases using SGD with momentum
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    // Compute velocity update
                    velocityWeights[j, i] = momentum * velocityWeights[j, i] + learningRate * gradientWeights[j, i];
                    // Update weights using momentum
                    weights[j, i] -= velocityWeights[j, i];
                    // Reset gradient
                    gradientWeights[j, i] = 0;
                }
                // Compute velocity update for biases
                velocityBiases[i] = momentum * velocityBiases[i] + learningRate * gradientBiases[i];
                // Update biases using momentum
                biases[i] -= velocityBiases[i];
                // Reset gradient
                gradientBiases[i] = 0;
            }
        }

        int ILayer.LayerType()
        {
            return 2;
        }

        double[,,] ILayer.Forward(double[,,] input)
        {
            throw new NotImplementedException();
        }

        double[,,] ILayer.Backward(double[,,] dLoss_dY, double learningRate, double momentum)
        {
            throw new NotImplementedException();
        }

        double[] ILayer.TransfromForward(double[,,] input)
        {
            throw new NotImplementedException();
        }

        double[,,] ILayer.TransfromBackward(double[] dLoss_dY)
        {
            throw new NotImplementedException();
        }
    }
}
