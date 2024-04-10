
using CNN.DataHandler;
using CNN.Layers;
using CNN.Model;

CifarSet cifarSet = new CifarSet("C:\\cifar-100-binary");

//simple model for testing
Model simpleCNN = new Model(13, 0.00000001, 0.01);
simpleCNN.AddLayer(new ConvolutionLayer(3, 32, 32, 3, 1, 10));
simpleCNN.AddLayer(new ReLuLayer3D(10, 30, 30));
simpleCNN.AddLayer(new MaxPoolLayer(10, 30, 30, 2, 2));
simpleCNN.AddLayer(new ConvolutionLayer(10, 15, 15, 4, 1, 20));
simpleCNN.AddLayer(new ReLuLayer3D(20, 12, 12));
simpleCNN.AddLayer(new MaxPoolLayer(20, 12, 12, 2, 2));
simpleCNN.AddLayer(new FlatLayer(20, 6, 6));
simpleCNN.AddLayer(new DenseLayer(720, 720));
simpleCNN.AddLayer(new ReLuLayer(720));
simpleCNN.AddLayer(new DenseLayer(720, 360));
simpleCNN.AddLayer(new ReLuLayer(360));
simpleCNN.AddLayer(new DenseLayer(360, 20));
simpleCNN.AddLayer(new SoftMaxLayer(20));

//accuracy prior to training
double[] results = new double[2];
results = simpleCNN.Test(cifarSet);
Console.WriteLine(results[0] + " "+ results[1]);

simpleCNN.Train(cifarSet, 1); // train the model
results = simpleCNN.Test(cifarSet); //recompute the accuracy
Console.WriteLine("Top one accuracy:" + results[0] + " Top three accuray" + results[1]);
