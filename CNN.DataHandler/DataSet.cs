using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace CNN.DataHandler
{
    internal class DataSet
    {
        private string baseDir;
        private string trainPath;
        private string testPath;
        private string fineLabelPath;
        

        public DataSet(string path) 
        { 
            this.baseDir = path;
            this.trainPath = Path.GetFullPath(baseDir + "\\train.bin");
            this.testPath = Path.GetFullPath(baseDir + "\\test.bin");
            this.fineLabelPath = Path.GetFullPath(baseDir + "\\fine_label_names.txt");
            byte[] bytesTrain = File.ReadAllBytes(trainPath);

            byte[] bytesTest = File.ReadAllBytes(testPath);
            System.Console.WriteLine(bytesTest[0]); System.Console.WriteLine(bytesTest[1]);
            for (int i =  2; i < 3074; i++) 
            { System.Console.Write(bytesTest[i]);}

            string[] fineLabels = File.ReadAllLines(fineLabelPath);
            System.Console.WriteLine(fineLabels[1]);
        }

    }
}
