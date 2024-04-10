// See https://aka.ms/new-console-template for more informationž
using System;
using CNN.DataHandler;

 CifarSet cifarSet = new CifarSet("C:\\cifar-100-binary");

for (int i = 0; i < 10; i++)
{
    Console.WriteLine(cifarSet.coarseLabels[cifarSet.trainDataLabel[i, 0]] + "   " + cifarSet.fineLabels[cifarSet.trainDataLabel[i, 1]]);
}


