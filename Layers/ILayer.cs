using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.Layers
{
    public interface ILayer
    {
        public int LayerType();
        public double[] Forward(double[] input);
        public double[,,] Forward(double[,,] input);
        public double[] Backward(double[] dLoss_dY, double learningRate, double momentum);
        public double[,,] Backward(double[,,] dLoss_dY, double learningRate, double momentum);
        public double[] TransfromForward(double[,,] input);
        public double[,,] TransfromBackward(double[] dLoss_dY);
    }
}
