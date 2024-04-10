using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    [Serializable]
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

        //loss wrt to input when using categorial cross entropy and softmax activation function
        public double[] Backward(double[] trueLabel, double learningRate, double momentum)
        {
            double[] gradient = new double[size];

            for (int i = 0; i < size; i++)
            {
                gradient[i] = output[i] - trueLabel[i];
            }
            
            return gradient;
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
