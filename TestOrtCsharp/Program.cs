using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class Program
{
    static void Main(string[] args)
    {
        // See https://aka.ms/new-console-template for more information
        Console.WriteLine("Hello, World!");

        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.EnableMemoryPattern = false;
        sessionOptions.AppendExecutionProvider_DML();
        //sessionOptions.AppendExecutionProvider_CPU();

        var session = new InferenceSession("models/add_fp32.onnx", sessionOptions);

        // create CPU input tensors
        int[] shape = { 3 };

        var aBuffer = new float[] { 0.12341f, 0.32f, -0.05f };
        var aTensor = new DenseTensor<float>(aBuffer, shape);

        var bBuffer = new float[] { 0.05124f, -0.12f, -0.0345f };
        var bTensor = new DenseTensor<float>(bBuffer, shape);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor<float>("A", aTensor),
            NamedOnnxValue.CreateFromTensor<float>("B", bTensor),
        };

        var output = session.Run(inputs);

        var yTensor = output.ToList().First().Value as DenseTensor<float>;

        for (int i = 0; i < shape[0]; i++)
        {
            Console.WriteLine(yTensor[i]);
        }
    }
}