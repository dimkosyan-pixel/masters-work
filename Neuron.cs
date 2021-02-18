using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackpropagation
{
    public class Neuron
    //реализация нейрона одного
    {
        public List<double> Weights { get; } // список весов
        public List<double> Inputs { get; } // список входных сигналов
        public NeuronType NeuronType { get; } // 
        public double Output { get; private set; } // выходой сигнад
        public double Delta { get; private set; } // переменная для промежуточного сохранения

        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        //inputCount колич входных нейронов
        {
            NeuronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();

            InitWeightsRandomValue(inputCount);
        }
        // инициализация весов рандомными значениями
        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            //сохранение входных сигналов
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }

            return Output;
        }
        // сигмоидная функция
        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            //производная сигмоидной функции
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        public void Learn(double error, double learningRate)// значение, влияющее на скорость обучения
        {
            //метод для вычисления новых весов
            if (NeuronType == NeuronType.Input)
            //если входной не обучаем
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeigth = weight - input * Delta * learningRate;
                //новый вес
                Weights[i] = newWeigth;
            }
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
