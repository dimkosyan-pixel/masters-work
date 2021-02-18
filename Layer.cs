using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNBackpropagation
{
   public class Layer
    //нейроны обьединяем в слои
    {
        public List<Neuron> Neurons { get; }

        public int NeuronCount => Neurons?.Count ?? 0;
        // вычмсляемое своиство дающее количество нейронов
        public NeuronType Type;

        public Layer(List<Neuron> neurons, NeuronType type = NeuronType.Normal)
        //конструктор(список нейронов котор проверяем на корректность и тип)
        {
            // TODO: проверить все входные нейроны на соответствие типу

            Neurons = neurons;
            Type = type;
        }

        public List<double> GetSignals()
        {
            //собираем все сигналы со слоя
            var result = new List<double>();
            foreach (var neuron in Neurons)
            {
                result.Add(neuron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
