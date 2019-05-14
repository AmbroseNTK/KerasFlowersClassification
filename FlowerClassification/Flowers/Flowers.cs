using Microsoft.ML.Scoring;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;

namespace FlowerClassification
{
    public partial class Flowers
    {
        const string modelName = "Flowers";
        private ModelManager manager;

        private static List<long> evaluateInput0ShapeForSingleInput = new List<long> { 1, 256, 256, 3 };
        private static List<string> evaluateInputNames = new List<string> { "input0" };
        private static List<string> evaluateOutputNames = new List<string> { "output0" };

        /// <summary>
        /// Returns an instance of Flowers model.
        /// </summary>
        public Flowers()
        {
            string codeBase = Assembly.GetExecutingAssembly().CodeBase;
            UriBuilder uri = new UriBuilder(codeBase);
            string dllpath = Uri.UnescapeDataString(uri.Path);
            string modelpath = Path.Combine(Path.GetDirectoryName(dllpath), "Flowers");
            string path = Path.Combine(modelpath, "00000001");
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Returns instance of Flowers model instantiated from exported model path.
        /// </summary>
        /// <param name="path">Exported model directory.</param>
        public Flowers(string path)
        {
            manager = new ModelManager(path, true);
            manager.InitModel(modelName, int.MaxValue);
        }

        /// <summary>
        /// Runs inference on Flowers model for a single input data.
        /// </summary>
        /// <param name="input0">; Shape of the input: { 1, 256, 256, 3 }</param>
        public IEnumerable<float> Evaluate(IEnumerable<float> input0)
        {
            List<Tensor> result = manager.RunModel(
                modelName,
                int.MaxValue,
                evaluateInputNames,
                new List<Tensor> { new Tensor(input0.ToList(), evaluateInput0ShapeForSingleInput) },
                evaluateOutputNames
            );

            List<float> r0 = new List<float>();
            result[0].CopyTo(r0);
            return r0;
        }
    } // END OF CLASS
} // END OF NAMESPACE
