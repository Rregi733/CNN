using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class ReLuLayer3D : ILayer
    {
        double[,,] input;
        double[,,] output;
        int z;
        int x;
        int y;
        public ReLuLayer3D(int z,int x, int y)
        {
            this.z = z;
            this.x = x;
            this.y = y;
            input = new double[z,x,y];
            output = new double[z, x, y];
        }

        public double[,,] Forward(double[,,] input)
        {
            this.input = input;

            for (int k = 0; k < z; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for(int n = 0; n < y; n++)
                    {
                        if (input[k, m, n] > 0)
                        {
                            output[k,m,n] = input[k, m, n];
                        }
                        else
                        {
                            output[k, m, n] = 0;
                        }
                    }
                }
            }

            return output;
        }

        public double[,,] Backward(double[,,] dLoss_dY, double learningRate, double momentum)
        {
            double[,,] gradient = new double[z,x,y];

            for (int k = 0; k < z; k++)
            {
                for (int m = 0; m < x; m++)
                {
                    for (int n = 0; n < y; n++)
                    {
                        if (input[k, m, n] > 0)
                        {
                            gradient[k, m, n] = dLoss_dY[k, m, n];
                        }
                        else
                        {
                            gradient[k, m, n] = 0;
                        }
                    }

                }
            }

            return gradient;
        }
    }
}
