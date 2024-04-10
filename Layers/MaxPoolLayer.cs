using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    [Serializable]
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

            outputWidth = inputWidth / poolWidth;
            outputHeight = inputHeight / poolHeight;

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

            //initialize mask with 0
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

                                // assign the first element as max
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

            for(int k = 0; k < dLoss_dY.GetLength(0); k++ )
            {
                for(int i = 0; i < dLoss_dY.GetLength(1); i++)
                {
                    for(int j = 0; j  < dLoss_dY.GetLength(2); j++)
                    {
                        for(int w = 0; w < poolWidth; w++)
                        {
                            for (int h = 0; h < poolHeight; h++)
                            {
                                //pass the gradient to elements that had maxvalue in previos layer
                                int inputX = poolWidth * i + w;
                                int inputY = poolHeight * j + h;
                                if (maxMask[k, inputX, inputY] == 1)
                                {
                                    inputGradient[k, inputX, inputY ] = dLoss_dY[k, i, j];
                                }
                                
                                else
                                {
                                    inputGradient[k, inputX, inputY] = 0;
                                }
                            }
                        }
                            
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
