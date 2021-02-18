using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackpropagation
{
    public class Topology
    //это набор своиств определяющую своиства нейронной сети
    {
        public int InputCount { get; }
        //входной слой

        public int OutputCount { get; }
        //коллич выходов
        public double LearningRate { get; }

        public List<int> HiddenLayers { get; }
        //количество скрытых слоев(для каждого слоя будет коллекция содерж
        //число нейронов на каждом слое)

        public Topology(int inputCount, int outputCount, double learningRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LearningRate = learningRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
