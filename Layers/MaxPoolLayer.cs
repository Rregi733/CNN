using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public class MaxPoolLayer : ILayer
    {
        //parameters of the input
        private int inputDepth;
        private int inputWidth;
        private int inputHeight;
        private double[,,] input;

        private int poolWidth;
        private int poolHeight;

        private int[,,] maxMask;
        private int outputWidth, outputHeight;
        private double[,,] output;
        public MaxPoolLayer(int inputDepth, int inputWidth, int inputHeight, int poolWidth, int poolHeight)
        {
            this.inputDepth = inputDepth;
            this.inputWidth = inputWidth;
            this.inputHeight = inputHeight;
            this.poolWidth = poolWidth;
            this.poolHeight = poolHeight;

            input = new double[inputDepth, inputWidth, inputHeight];

            outputWidth = inputWidth - poolWidth + 1;
            outputHeight = inputHeight - poolHeight + 1;

            output = new double[inputDepth, outputWidth, outputHeight];

            maxMask = new int[inputDepth, inputWidth, inputHeight];

            for(int k = 0;k < inputDepth; k++)
            {
                for(int i = 0;i < inputWidth;i++)
                {
                    for(int j = 0;j < inputHeight;j++)
                    {
                        maxMask[k, i, j] = 0;
                    }
                }
            }
        }

        public double[,,] Forward( double[,,] input)
        {
            this.input= input;

            for (int k = 0; k < inputDepth; k++)
            {
                for (int i = 0; i < inputWidth; i++)
                {
                    for (int j = 0; j < inputHeight; j++)
                    {
                        maxMask[k, i, j] = 0;
                    }
                }
            }

            for (int k = 0; k < inputDepth; k++)
            {
                for(int i = 0; i < outputWidth;i++)
                {
                    for(int j = 0; j < outputHeight;j++)
                    {
                        double maxValue = 0;
                        int maxX = 0, maxY = 0;
                        for(int m  = 0; m < poolWidth;m++)
                        {                            
                            for(int n = 0; n < poolHeight; n++)
                            {
                                int inputX = i * poolWidth + m;
                                int inputY = j * poolHeight + n;

                                if(m == 0 && n == 0)
                                {
                                    maxValue = input[k, inputX, inputY];
                                    maxX = inputX; maxY = inputY;
                                }
                                else if(input[k, inputX, inputY] > maxValue)
                                {
                                    maxValue = input[k, inputX, inputY];
                                    maxX = inputX; maxY = inputY;
                                }
                            }

                        }
                        output[k, i, j] = maxValue;
                        maxMask[k, maxX, maxY] = 1;
                    }
                }
            }
            return output;
        }

        public double[,,] Backward(double[,,] dLoss_dY, double learningRate, double momentum)
        {
            double[,,] inputGradient = new double[inputDepth, inputWidth, inputHeight];

            for(int k = 0; k < inputDepth; k++ )
            {
                for(int i = 0; i < inputWidth; i++)
                {
                    for(int j = 0; j  < inputHeight; j++)
                    {
                        inputGradient[k, i, j] = dLoss_dY[k, i, j] * maxMask[k, i, j];
                    }
                }
            }

            return inputGradient;
        }

        int ILayer.LayerType()
        {
            return 1;
        }

        double[] ILayer.Forward(double[] input)
        {
            throw new NotImplementedException();
        }

        double[] ILayer.Backward(double[] dLoss_dY, double learningRate, double momentum)
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
