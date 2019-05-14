using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.Storage;
using Windows.AI.MachineLearning;
using Windows.Foundation;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;
using Windows.Storage.Streams;
using Windows.Graphics.Imaging;
using Windows.UI;

namespace FlowersClassificationUWP
{
    class ModelUtils
    {
        /// <summary>
        /// Input width of model, do not change this!!!
        /// </summary>
        const int INPUT_WIDTH = 256;
        /// <summary>
        /// Input heigth of model, do not change this!!!
        /// </summary>
        const int INPUT_HEIGHT = 256;
        /// <summary>
        /// Current model
        /// </summary>
        public Inceptionv3_convertedModel Model;
        private ModelUtils()
        {

        }
        private static ModelUtils instance;
        /// <summary>
        /// Singleton pattern
        /// </summary>
        public static ModelUtils Utils
        {
            get
            {
                if(instance == null)
                {
                    instance = new ModelUtils();
                }
                return instance;
            }
        }
        public async Task LoadModel()
        {
            var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri("ms-appx:///Assets/inceptionv3_converted.onnx"));
            Model = await Inceptionv3_convertedModel.CreateFromStreamAsync(modelFile);
        }

        public async Task<Inceptionv3_convertedOutput> Evaluate(StorageFile file)
        {
            Inceptionv3_convertedInput tensorInput = new Inceptionv3_convertedInput();
            byte[] image = await ResizedImage(file, INPUT_WIDTH, INPUT_HEIGHT);
           
            
            List<float> input = new List<float>();
            List<float> R = new List<float>();
            List<float> G = new List<float>();
            List<float> B = new List<float>();
            for (int j = 0; j < INPUT_HEIGHT; j++)
            {
                for (int i = 0; i < INPUT_WIDTH; i++)
                {
                   
                    R.Add(GetPixel(image,i, j,INPUT_WIDTH,INPUT_HEIGHT).R / 255f);
                    G.Add(GetPixel(image, i, j, INPUT_WIDTH, INPUT_HEIGHT).G / 255f);
                    B.Add(GetPixel(image, i, j, INPUT_WIDTH, INPUT_HEIGHT).B / 255f);
                }
            }
            input.AddRange(R);
            input.AddRange(G);
            input.AddRange(B);
            tensorInput.input_1_0 = TensorFloat.CreateFromArray(new long[] { 1, 256, 256, 3 },input.ToArray());
            return await Model.EvaluateAsync(tensorInput);
        }
        public static async Task<Byte[]> ResizedImage(StorageFile imageFile, int maxWidth, int maxHeight)
        {
            IRandomAccessStream inputstream = await imageFile.OpenReadAsync();
           
            BitmapDecoder decoder = await BitmapDecoder.CreateAsync(inputstream);
            BitmapTransform transform = new BitmapTransform();
            transform.ScaledWidth = INPUT_WIDTH;
            transform.ScaledHeight = INPUT_HEIGHT;

            var data = await decoder.GetPixelDataAsync(BitmapPixelFormat.Rgba8, BitmapAlphaMode.Ignore, transform, ExifOrientationMode.IgnoreExifOrientation, ColorManagementMode.DoNotColorManage);
            var bytes = data.DetachPixelData();

            return bytes;
        }
        public Color GetPixel(byte[] pixels, int x, int y, uint width, uint height)
        {
            int i = x;
            int j = y;
            int k = (i * (int)width + j) * 3;
            var r = pixels[k + 0];
            var g = pixels[k + 1];
            var b = pixels[k + 2];
            return Color.FromArgb(0, r, g, b);
        }
    }
}
