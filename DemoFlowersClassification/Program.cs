using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FlowerClassification;
using System.Drawing;

namespace DemoFlowersClassification
{
    class Program
    {
        static FlowerInceptionV3 model;
        const int INPUT_WIDTH = 256;
        const int INPUT_HEIGHT = 256;
        static void Main(string[] args)
        {
            model = new FlowerInceptionV3();
            Bitmap image = new Bitmap(Image.FromFile(@"D:\Repos\KerasFlowersClassification\dataset\flowers\test\rose\16152205512_9d6cb80fb6.jpg"), new Size(INPUT_WIDTH,INPUT_HEIGHT));
            List<float> input = new List<float>();
            List<float> R = new List<float>();
            List<float> G = new List<float>();
            List<float> B = new List<float>();
            for (int j = 0; j < INPUT_HEIGHT; j++)
            {
                for(int i = 0; i < INPUT_WIDTH; i++)
                {
                    R.Add(image.GetPixel(i, j).R/255f);
                    G.Add(image.GetPixel(i, j).G/255f);
                    B.Add(image.GetPixel(i, j).B/255f);
                }
            }
            input.AddRange(R);
            input.AddRange(G);
            input.AddRange(B);
            IEnumerable<float> result = model.Evaluate(input);
            foreach(float p in result)
            {
                Console.WriteLine(p);
            }
            Console.Read();
            
        }
    }
}
