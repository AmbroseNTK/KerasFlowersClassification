// This file was automatically generated by VS extension Windows Machine Learning Code Generator VS 2017 v2.6
// from model file inceptionv3_converted.onnx
// Warning: This file may get overwritten if you add add an onnx file with the same name
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Streams;
using Windows.AI.MachineLearning;
namespace FlowersClassificationUWP
{
    
    public sealed class Inceptionv3_convertedInput
    {
        public TensorFloat input_1_0; // shape(1,256,256,3)
    }
    
    public sealed class Inceptionv3_convertedOutput
    {
        public TensorFloat dense_2_Softmax_01; // shape(1,5)
    }
    
    public sealed class Inceptionv3_convertedModel
    {
        private LearningModel model;
        private LearningModelSession session;
        private LearningModelBinding binding;
        public static async Task<Inceptionv3_convertedModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
        {
            Inceptionv3_convertedModel learningModel = new Inceptionv3_convertedModel();
            learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
            learningModel.session = new LearningModelSession(learningModel.model);
            learningModel.binding = new LearningModelBinding(learningModel.session);
            return learningModel;
        }
        public async Task<Inceptionv3_convertedOutput> EvaluateAsync(Inceptionv3_convertedInput input)
        {
            binding.Bind("input_1_0", input.input_1_0);
            var result = await session.EvaluateAsync(binding, "0");
            var output = new Inceptionv3_convertedOutput();
            output.dense_2_Softmax_01 = result.Outputs["dense_2_Softmax_01"] as TensorFloat;
            return output;
        }
    }
}
