using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Net.Mime;

public class Program
{
    static void RunModel<T>(DenseTensor<T> aTensor, DenseTensor<T> bTensor, int[] shape, bool useCpu = false)
    {
        var sessionOptions = new SessionOptions();
        sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        sessionOptions.EnableMemoryPattern = false;
        if (useCpu)
        {
            sessionOptions.AppendExecutionProvider_CPU();
        }
        else
        {
            sessionOptions.AppendExecutionProvider_DML();
        }

        var session = new InferenceSession("models/add_fp32.onnx", sessionOptions);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor<T>("A", aTensor),
            NamedOnnxValue.CreateFromTensor<T>("B", bTensor),
        };

        var output = session.Run(inputs);

        var yTensor = output.ToList().First().Value as DenseTensor<float>;

        for (int i = 0; i < shape[0]; i++)
        {
            Console.WriteLine(yTensor[i]);
        }
    }

    static void Main(string[] args)
    {
        bool useFp16 = args.Length > 0 && args[0] == "fp16";

        int[] shape = { 3 };
        var aBuffer = new float[] { 0.12341f, 0.32f, -0.05f };
        var bBuffer = new float[] { 0.05124f, -0.12f, -0.0345f };

        if (useFp16)
        {
        }
        else
        {
            var aTensor = new DenseTensor<float>(aBuffer, shape);
            var bTensor = new DenseTensor<float>(bBuffer, shape);
            RunModel<float>(aTensor, bTensor, shape);
        }
    }
}