using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using CNN.Layers;
using CNN.DataHandler;
using System.Data;

namespace CNN.Model
{
    [Serializable]
    public class Model
    {
        ILayer[] layers;
        int layerCounter = 0;
        double learningRate;
        double momentum;
        public Model(int nrLayers, double learningRate, double momentum)
        {
            layers = new ILayer[nrLayers];
            this.learningRate = learningRate;
            this.momentum = momentum;
        }

        public void AddLayer(ILayer layer)
        {
            this.layers[layerCounter] = layer;
            layerCounter++;
        }

        public double[] Forward(double[,,] input)
        {
            //count the layers and their types in order to allocate space for intermediete arrays
            int volLayersCount = 0;
            int flatLayersCount = 0;
            for (int i = 0; i < this.layers.Length; i++)
            {
                if (layers[i].LayerType() == 1)
                {
                    volLayersCount++;
                }
                else if (layers[i].LayerType() == 2)
                {
                    flatLayersCount++;
                }
                else if (layers[i].LayerType() == 3)
                {
                    flatLayersCount++;
                }
            }

            double[][,,] interVol = new double[volLayersCount][,,];
            int v = 0;

            double[][] interFlat = new double[flatLayersCount][];
            int f = 0;

            interVol[v] = layers[0].Forward(input);
            v++;

            // run the forward methods of each layer
            for (int i = 1; i < this.layers.Length; i++)
            {
                if (layers[i].LayerType() == 1)
                {
                    interVol[v] = layers[i].Forward(interVol[v-1]);
                    v++;
                }
                else if (layers[i].LayerType() == 2)
                {
                    interFlat[f] = layers[i].Forward(interFlat[f-1]);
                    f++;
                }
                else if (layers[i].LayerType() == 3)
                {
                    interFlat[f] = layers[i].TransfromForward(interVol[v-1]);
                    f++;
                }
            }

            return interFlat[f-1];
        }

        //count the layers and their types in order to allocate space for intermediete arrays
        public void Backward(double[] trueLabel, double learningRate, double momentum)
        {
            int volLayersCount = 0;
            int flatLayersCount = 0;
            for (int i = 0; i < this.layers.Length; i++)
            {
                if (layers[i].LayerType() == 1)
                {
                    volLayersCount++;
                }
                else if (layers[i].LayerType() == 2)
                {
                    flatLayersCount++;
                }
                else if (layers[i].LayerType() == 3)
                {
                    volLayersCount++;
                }
            }

            double[][,,] interVol = new double[volLayersCount][,,];
            int v = 0;

            double[][] interFlat = new double[flatLayersCount][];
            int f = 0;

            interFlat[f] = layers[this.layers.Length - 1].Backward(trueLabel, learningRate, momentum);
            f++;

            // run the backward methods of each layer
            for (int i = this.layers.Length - 2; i >= 0; i--)
            {
                if (layers[i].LayerType() == 1)
                {
                    interVol[v] = layers[i].Backward(interVol[v-1], learningRate, momentum);
                    v++;
                }
                else if (layers[i].LayerType() == 2)
                {
                    interFlat[f] = layers[i].Backward(interFlat[f - 1], learningRate, momentum);
                    f++;
                }
                else if (layers[i].LayerType() == 3)
                {
                    interVol[v] = layers[i].TransfromBackward(interFlat[f - 1]);
                    v++;
                }
            }

        }

        public void Train(CNN.DataHandler.CifarSet cifarSet, int epoch)
        {
            
                //Do a forward and a backward pass for each image, for the number of epochs
                for (int i = 0; i < epoch; i++)
                {
                    // assign input image
                    double[,,] image = new double[3, 32, 32];
                    for (int c = 0; c < 3; c++)
                    {
                        for (int h = 0; h < 32; h++)
                        {
                            for (int w = 0; w < 32; w++)
                            {
                                image[c, h, w] = cifarSet.trainData[i, c, h, w];
                            }
                        }
                    }

                    //Forward pass
                    double[] output = this.Forward(image);

                    //Onehot encoding true label
                    double[] trueLabel = new double[output.Length];
                    for (int j = 0; j < output.Length; j++)
                    {
                        if (cifarSet.trainDataLabel[i, 0] == j) //0 - coarse label 1 - fine label
                        {
                            trueLabel[j] = 1;
                        }
                        else
                        {
                            trueLabel[j] = 0;
                        }
                    }

                    //Backward pass
                    this.Backward(trueLabel, learningRate, momentum);
                    Console.WriteLine(i);
                }
            
            
        }

        //Function to find top elements in an array, used in testing phase
        static int FindIndexOfMax(double[] numbers)
        {
            if (numbers == null || numbers.Length == 0)
                throw new ArgumentException("Array cannot be null or empty");

            int maxIndex = 0;

            for (int i = 1; i < numbers.Length; i++)
            {
                if (numbers[i] > numbers[maxIndex])
                {
                    maxIndex = i;
                }
            }

            return maxIndex;
        }

        static int[] FindIndicesOfMaxThree(double[] numbers)
        {
            int[] indices = Enumerable.Range(0, numbers.Length).ToArray();

            // Sort indices based on corresponding element values
            Array.Sort(indices, (a, b) => numbers[b].CompareTo(numbers[a]));

            // Extract the indices of the top three elements
            int[] maxThreeIndices = new int[3];
            Array.Copy(indices, maxThreeIndices, 3);

            return maxThreeIndices;
        }
        public double[] Test(CNN.DataHandler.CifarSet cifarSet)
        {
            int topOneHit = 0;
            int topThreeHit = 0;
            int testSetSize = cifarSet.testDataLabel.GetLength(0);

            //run the tests for each image in the test set
            for (int i = 0;i < 300;i++)
            {
                // assign input image
                double[,,] image = new double[3, 32, 32];
                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < 32; h++)
                    {
                        for (int w = 0; w < 32; w++)
                        {
                            image[c, h, w] = cifarSet.trainData[i, c, h, w];
                        }
                    }
                }

                // run forward pass on model and retrive the output
                double[] output = this.Forward(image);

                // compare the predictions against the test label for the image
                int maxIndex = FindIndexOfMax(output);
                if (maxIndex == cifarSet.trainDataLabel[i, 0]) { topOneHit++; } //0 - coarse label 1 - fine label

                int[] maxThreeIndices = FindIndicesOfMaxThree(output);
                for(int m = 0; m < 3; m++)
                {
                    if (maxThreeIndices[m] == cifarSet.trainDataLabel[i, 0]) { topThreeHit++; } //0 - coarse label 1 - fine label
                }
            }

            double topOneAccuracy = (double)topOneHit / 300 ;
            double topThreeAccuracy = (double)topThreeHit / 300 ;
            double[] result = { topOneAccuracy, topThreeAccuracy };

            return result;
        }


    }
}
