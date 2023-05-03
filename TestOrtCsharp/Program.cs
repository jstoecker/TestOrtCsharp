using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Net.Mime;

public class Program
{
    static void PrintTensor(DenseTensor<Float16> tensor, int[] shape)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            var val = tensor[i].value;
            var halfVal = BitConverter.UInt16BitsToHalf(val);
            Console.WriteLine(halfVal);
        }
    }

    static void PrintTensor(DenseTensor<float> tensor, int[] shape)
    {
        for (int i = 0; i < shape[0]; i++)
        {
            Console.WriteLine(tensor[i]);
        }
    }

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

        var useFp16 = typeof(T) == typeof(Float16);

        var modelPath = useFp16 ? "models/add_fp16.onnx" : "models/add_fp32.onnx";

        var session = new InferenceSession(modelPath, sessionOptions);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor<T>("A", aTensor),
            NamedOnnxValue.CreateFromTensor<T>("B", bTensor),
        };

        var output = session.Run(inputs);

        if (useFp16)
        {
            PrintTensor(output.ToList().First().Value as DenseTensor<Float16>, shape);
        }
        else
        {
            PrintTensor(output.ToList().First().Value as DenseTensor<float>, shape);
        }
    }

    static void RunTest(bool useFp16)
    {
        int[] shape = { 6 };
        var aBuffer = new float[] { 0.12341f, 0.32f, -0.05f, 1, 3, 11 };
        var bBuffer = new float[] { 0.05124f, -0.12f, -0.0345f, 2, -1, -5 };

        if (useFp16)
        {
            var aTensor = new DenseTensor<Float16>(shape);
            var bTensor = new DenseTensor<Float16>(shape);
            for (int i = 0; i < shape[0]; i++)
            {
                aTensor[i] = BitConverter.HalfToUInt16Bits((Half)aBuffer[i]);
                bTensor[i] = BitConverter.HalfToUInt16Bits((Half)bBuffer[i]);
            }

            RunModel<Float16>(aTensor, bTensor, shape);
        }
        else
        {
            var aTensor = new DenseTensor<float>(aBuffer, shape);
            var bTensor = new DenseTensor<float>(bBuffer, shape);
            RunModel<float>(aTensor, bTensor, shape);
        }
    }

    static void Main(string[] args)
    {
        Console.WriteLine("RUN WITH FP32:");
        RunTest(false);
        Console.WriteLine("RUN WITH FP16:");
        RunTest(true);
    }
}