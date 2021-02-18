using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackpropagation
{
    public class NeuralNetwork
    //реализация нейронной сети(это коллекция слоев)
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }


        // метод для обучения
        public double Learn(List<Tuple<double, double[]>> dataset, int epoch)//структура данных

        {
            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {

                foreach (var data in dataset)
                {
                    error += Backpropagation(data.Item1, data.Item2);
                }
            }
            var result = error / epoch;//вычисление средней ошибки
            return result;
        }
        private double Backpropagation(double exprected, params double[] inputs)
        {
            //обратное распространение ошибки
            var actual = FeedForward(inputs).Output;

            //double exprected-тот результат который мы ожидаем
            //params double[] inputs входные сигналы
            //метод обратного распространения ошибки, expected-ожидаемый сигнал

            var difference = actual - exprected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            //для других слоев, двидение справа на лево, -2 один слой уже обучили
            for (int j = Layers.Count - 2; j >= 0; j--)
            {
                var layer = Layers[j];
                var previousLayer = Layers[j + 1];
                //обучаем послойно
                for (int i = 0; i < layer.NeuronCount; i++)
                // переберает нейроны в одном слое
                {
                    var neuron = layer.Neurons[i];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    // 
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        //обучаем текущий нейрон
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            // возвращение квадратичной ошибки
            var result = difference * difference;
            return result;
        }
        //public Neuron FeedForward(List<double> inputSignals)
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeurons(inputSignals.ToArray());
            FeedForwardAllLayersAfterInput();
            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();

        }
        private void FeedForwardAllLayersAfterInput()
        //прогонка для всех нейронов
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSingals = Layers[i - 1].GetSignals();
                // перебераем все нейроны этого слоя и отправляем сигналы с предыд слоя
                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSingals);
                }
            }
        }

        private void SendSignalsToInputNeurons(params double[] inputSignals)
        {
            //передача сигнала с одного слоя на другой
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }

        private void CreateOutputLayer()
        {//выходной слой
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                //lastLayer.NeuronCount-колич нейронов на предыд слое
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        private void CreateHiddenLayers()
        //генерация скрытых слоев     
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        private void CreateInputLayer()
        {
            //создаем входной слой
            var inputNeurons = new List<Neuron>();
            // количество входных нейронов хранится в топологии
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                // у входных нейронов вход всегда 1
                inputNeurons.Add(neuron);
            }
            //cоздаем слой
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
    }
}
