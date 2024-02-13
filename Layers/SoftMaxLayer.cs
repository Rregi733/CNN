using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class SoftMaxLayer : ILayer
    {
        double[] input;
        double[] output;
        int size;
        public SoftMaxLayer(int size) 
        {
            this.size = size;
            input = new double[size];
            output = new double[size];
        }

        public double[] Forward(double[] input)
        {
            this.input = input;
            double expSum = 0.0;
            double maxElement = input.Max();
            for(int i = 0; i < size; i++)
            {
                output[i] = Math.Exp(input[i] - maxElement); // subtructing max element for numerical stability
                expSum += output[i];
            }

            for(int i = 0;i < size; i++)
            {
                // normalizing the output
                output[i] = output[i]/expSum;
            }
            return output;
        }

        public double[] Backward(double[] trueLabel, double learningRate)
        {
            double[] gradient = new double[size];

            for (int i = 0; i <= size; i++)
            {
                gradient[i] = output[i] - trueLabel[i];
            }
            
            return gradient;
        }
    }
}
