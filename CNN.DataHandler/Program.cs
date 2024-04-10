// See https://aka.ms/new-console-template for more informationž
using System;
using CNN.DataHandler;

 DataSet dataSet = new DataSet("C:\\cifar-100-binary");

for (int i = 0; i < 10; i++)
{
    Console.WriteLine(dataSet.coarseLabels[dataSet.trainDataLabel[i, 0]] + "   " + dataSet.fineLabels[dataSet.trainDataLabel[i, 1]]);
}


