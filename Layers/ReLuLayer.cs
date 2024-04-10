using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    [Serializable]
    public class ReLuLayer : ILayer
    {
        double[] input;
        double[] output;
        int size;
        public ReLuLayer(int size)
        {
            this.size = size;
            input = new double[size];
            output = new double[size];
        }

        //pass positive inputs and make others zero
        public double[] Forward(double[] input)
        {
            this.input = input;
            
            for (int i = 0; i < size; i++)
            {
                if (input[i] > 0)
                {
                    output[i] = input[i];
                }
                else
                {
                    output[i] = 0;
                }
            }

            return output;
        }

        //Pass gradients of the inputs that ware positive
        public double[] Backward(double[] dLoss_dY, double learningRate, double momentum)
        {
            double[] gradient = new double[size];

            for (int i = 0; i < size; i++)
            {
                if ((input[i] > 0))
                {
                    gradient[i] = dLoss_dY[i];
                }
                else
                {
                    gradient[i] = 0;
                }
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
