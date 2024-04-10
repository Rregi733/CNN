using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class FlatLayer : ILayer
    {
        double[,,] input;
        double[] output;
        int z;
        int x;
        int y;
        public FlatLayer(int depth,int x, int y) 
        {
            this.z = depth;
            this.x = x;
            this.y = y;
            this.input = new double[z,x,y];
            this.output = new double[x*y*z];
        }
        
        public double[] TransfromForward(double[,,] input)
        {
            this.input = input;
            int i = 0;
            for (int k = 0; k < z; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for (int n = 0; n < y; n++)
                    {

                            output[i] = input[k, m, n];
                            i++;
                    }
                }
            }
            return output;
        }

        public double[,,] TransfromBackward(double[] dLoss_dY)
        {
            double[,,] gradient = new double[z,x,y];
            int i = 0;
            for (int k = 0; k < z; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for (int n = 0; n < y; n++)
                    {
                        input[k, m, n] = dLoss_dY[i];
                        i++;
                    }
                }
            }
            return gradient;
        }

        int ILayer.LayerType()
        {
            return 3;
        }

        double[] ILayer.Forward(double[] input)
        {
            throw new NotImplementedException();
        }

        double[,,] ILayer.Forward(double[,,] input)
        {
            throw new NotImplementedException();
        }

        double[] ILayer.Backward(double[] dLoss_dY, double learningRate, double momentum)
        {
            throw new NotImplementedException();
        }

        double[,,] ILayer.Backward(double[,,] dLoss_dY, double learningRate, double momentum)
        {
            throw new NotImplementedException();
        }
    }
}
