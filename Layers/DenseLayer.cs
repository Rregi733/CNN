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

        public DenseLayer(int inputSize, int outputSize)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;

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
    }
}
