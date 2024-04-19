using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using CNN.Layers;
using CNN.DataHandler;
using System.Data;
using CNN.Model;
using System.Runtime.InteropServices;
using System.Reflection;

int mode = 0;
string dataPath = null;
bool dataPathExists = false;
DataSerializer dataSerializer = new DataSerializer();

//array to hold information about saved models
string[,] models =
{
    {"simple_model_1", "13", "0.0001","0.01" },
    {"simple_model_2", "13", "0.0001","0.00" },
    {"simple_model_3", "13", "0.0002","0.01" },
    {"simple_model_4", "13", "0.0002","0.00" }
};

//selecting the main tab of the program
while(mode != 1 && mode != 2)
{   
    Console.Clear();
    Console.WriteLine("Select mode:");
    Console.WriteLine("1 - Single model benchmark");
    Console.WriteLine("2 - Comparative tests");
    Console.WriteLine();
    string input = Console.ReadLine();
    Int32.TryParse(input, out mode);
}

// inputing the path to CIFAR 100 data set
while (!dataPathExists)
{
    Console.Clear();
    Console.WriteLine("Enter '1' for the defult path");
    Console.WriteLine("1 - Default Cifar 100 path: C:\\cifar-100-binary");
    Console.WriteLine("Or input the path manually:");
    Console.WriteLine();
    dataPath = Console.ReadLine();
    if(dataPath == "1")
    {
        dataPath = "C:\\cifar-100-binary";
    }

    if (Directory.Exists(dataPath))
    {
        dataPathExists = true;
    }
    else
    {
        Console.WriteLine("{0} is not a vaild directory.", dataPath);
    }
}

if (mode == 1)
{
    // inputs specific to benchmark mode
    int model = 0;
    int testMode = 0;

        // model selection
        while(!((model - 1) >= 0 && (model - 1) < models.GetLength(0)))
        {
            Console.Clear();
            Console.WriteLine("Select model:");
            for (int i = 0; i < models.GetLength(0); i++)
            {
                Console.WriteLine("{0} - Model name: {1} Layers: {2} Learning rate: {3} Momentum: {4}", i + 1, models[i, 0], models[i, 1], models[i, 2], models[i, 3]);
            }
            Console.WriteLine();
            string input = Console.ReadLine();
            Int32.TryParse(input, out model);
            if(!((model - 1) >= 0 && (model - 1) < models.GetLength(0)))
            {
                Console.WriteLine("Input not valid");
            }
        }

        // test type selection
        while(!(testMode == 1 || testMode == 2))
        {
            Console.Clear();
            Console.WriteLine("Select mode of testing:");
            Console.WriteLine("1 - Top one accuracy test");
            Console.WriteLine("2 - Top three accuracy test");
            Console.WriteLine();
            string input = Console.ReadLine();
            Int32.TryParse(input, out testMode);

            if (!(testMode == 1 || testMode == 2))
            {
                Console.WriteLine("Input not valid");
            }
        }

    CifarSet cifarSet = new CifarSet(dataPath); //load the Cifar set

    Model s = dataSerializer.BinaryDeserialize(models[(model - 1), 0]) as Model; //load the model to be tested

    double[] results = new double[2]; //create an array for the results
    results = s.Test(cifarSet); //compute the accuracy

    //Display the results
    Console.Clear();
    Console.WriteLine("Model name: {0}",models[(model - 1), 0]);
    if (testMode == 1)
    {
        Console.WriteLine("Top one accuracy: " + Math.Round(results[testMode - 1], 5));
    }
    else if (testMode == 2)
    {
        Console.WriteLine("Top three accuray: " + Math.Round(results[testMode - 1], 5));
    }
}
else
{
    //Displaing the results in comparative mode

    CifarSet cifarSet = new CifarSet(dataPath); //load the Cifar set
    double[] results = new double[2]; //create an array for the results
    Console.Clear();
    for (int i = 0; i < models.GetLength(0); i++)
    {
        Model s = dataSerializer.BinaryDeserialize(models[i, 0]) as Model; //load the model to be tested
        results = s.Test(cifarSet); //compute the accuracy

        // display info and results for the model
        Console.WriteLine("{0} - Model name: {1} Layers: {2} Learning rate: {3} Momentum: {4}", i + 1, models[i, 0], models[i, 1], models[i, 2], models[i, 3]);
        Console.WriteLine("Top one accuracy: {0} \tTop three accuray: {1}",Math.Round(results[0], 5), Math.Round(results[1], 5));
        Console.WriteLine();
    }
}
