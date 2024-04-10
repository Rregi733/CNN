using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace CNN.DataHandler
{
    public class CifarSet
    {
        private string baseDir;
        private string trainPath;
        private string testPath;
        private string coarseLabelPath;
        private string fineLabelPath;
        public string[] coarseLabels;
        public string[] fineLabels;

        public double[,,,] trainData = new double[50000, 3, 32, 32];
        public int[,] trainDataLabel = new int[50000, 2];

        public double[,,,] testData = new double[10000, 3, 32, 32];
        public int[,] testDataLabel = new int[50000, 2];

        public CifarSet(string path) 
        { 
            this.baseDir = path;
            this.trainPath = Path.GetFullPath(baseDir + "\\train.bin");
            this.testPath = Path.GetFullPath(baseDir + "\\test.bin");
            this.coarseLabelPath = Path.GetFullPath(baseDir + "\\coarse_label_names.txt");
            this.fineLabelPath = Path.GetFullPath(baseDir + "\\fine_label_names.txt");

            //Get the names corresponding to each layer
            this.coarseLabels = File.ReadAllLines(coarseLabelPath);
            this.fineLabels = File.ReadAllLines(fineLabelPath);

            //Populate the test data structure
            byte[] bytesTrain = File.ReadAllBytes(trainPath);
            for( int i = 0; i < 50000; i++)
            {
                int b = i * 3074;
                trainDataLabel[i, 0] = Convert.ToInt32(bytesTrain[b]);
                b++;
                trainDataLabel[i, 1] = Convert.ToInt32(bytesTrain[b]);
                b++;

                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < 32; h++)
                    {
                        for (int w = 0; w < 32; w++)
                        {
                            trainData[i, c, h, w] = Convert.ToDouble(bytesTrain[b]);
                            b++;
                        }
                    }
                }
            }

            //Populate the test data structure
            byte[] bytesTest = File.ReadAllBytes(testPath);
            for (int i = 0; i < 10000; i++)
            {
                int b = i * 3074;
                testDataLabel[i, 0] = Convert.ToInt32(bytesTest[b]);
                b++;
                testDataLabel[i, 1] = Convert.ToInt32(bytesTest[b]);
                b++;

                for (int c = 0; c < 3; c++)
                {
                    for (int h = 0; h < 32; h++)
                    {
                        for (int w = 0; w < 32; w++)
                        {
                            testData[i, c, h, w] = Convert.ToDouble(bytesTest[b]);
                            b++;
                        }
                    }
                }
            }


        }

    }
}
