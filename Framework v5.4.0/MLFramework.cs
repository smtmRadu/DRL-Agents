using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.IO;
using UnityEngine.UI;
using System.IO;
using System.Text;
using System.Linq;
using UnityEditor;
using TMPro;
using System;
using UnityEditor.ShortcutManagement;
using Unity.VisualScripting;
using System.ComponentModel;
using UnityEditor.TerrainTools;
using UnityEngine.UIElements;
using System.Collections.Concurrent;
using System.Runtime.Serialization;
using UnityEngine.Rendering;
using static UnityEditor.Progress;

namespace MLFramework
{
    // v5.4.3
    // +adds for sensor buffer
    // +one agent per environment update
    // +OnEpisodeBegin() updated
    // +training data file direct name added

    //5.4.2
    //softmax added
    public class NeuralNetwork
    {
        public static ActivationFunctionType activation = ActivationFunctionType.Tanh;
        public static ActivationFunctionType outputActivation = ActivationFunctionType.Tanh;
        public static MutationStrategy mutation = MutationStrategy.Classic;
        public static InitializationFunctionType initialization = InitializationFunctionType.StandardNormal1;

        protected int[] layers;
        protected float[][] neurons;
        protected float[][][] weights;
        protected float[][] biases;
        protected float fitness;

        public NeuralNetwork(int[] layers)
        {
            InitializeLayers(layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            fitness = 0f;

        }
        public NeuralNetwork(NeuralNetwork copyNN)
        {
            InitializeLayers(copyNN.layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            SetWeightsWith(copyNN.weights);
            SetBiasesWith(copyNN.biases);
            SetFitness(copyNN.GetFitness());
        }
        public NeuralNetwork(string path)
        {
            if (new FileInfo(path).Length == 0)
            {
                Debug.LogError("<color=red>The training cannot start! Reason: Brain Model uploaded file is empty</color>");
                return;
            }
            //JSON serialization doesn't work. Also, ToJson() serialize just public fields of the class.
            /* string json = System.IO.File.ReadAllText(path);
           NeuralNetwork copyNN = JsonUtility.FromJson<NeuralNetwork>(json);
           InitializeLayers(copyNN.layers);
           InitializeNeuronsAndBiases(false);
           InitializeWeights(false);
           SetWeightsWith(copyNN.weights);
           SetBiasesWith(copyNN.biases);
           SetFitness(copyNN.GetFitness());
           return;*/

            //For each line, there is 1 more element that was read when splitting, so kill it everywhere
            List<string> fileLines = File.ReadAllLines(path).ToList();

            //Get Layers Data
            string[] layersLineStr = fileLines[0].Split("n,");//one more read than neccesary
            int noLayers = layersLineStr.Length - 1;
            int[] layersLineInt = new int[noLayers];
            Functions.ConvertStrArrToIntArr(layersLineStr, ref layersLineInt);
            InitializeLayers(layersLineInt);
            InitializeNeuronsAndBiases(true);
            InitializeWeights(true);

            //Get Weights Data
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < noLayers; i++)//First and last lineStr are ignored ( first lineStr is layerFormat, last lineStr is Fitness)
            {
                //One lineStr here are the weightss on a single neuronsNumber
                List<float[]> weightsOnLayer = new List<float[]>();

                string[] lineStr = fileLines[i].Split("w,");// one more read than neccesary
                float[] lineInt = new float[lineStr.Length - 1];
                Functions.ConvertStrArrToFloatArr(lineStr, ref lineInt);


                //This probArr must be devided depeding on the previous neuronsNumber number of neurons
                int numNeurOnPrevLayer = layersLineInt[i - 1];
                float[] weightsOnNeuron = new float[numNeurOnPrevLayer];
                int count = 0;

                for (int j = 0; j < lineInt.Length; j++)
                {

                    ///Problema cand ajunge la line32.lenght -1, ultima adaugare nu o baga in weightss on Layer
                    if (count < numNeurOnPrevLayer)
                    {
                        weightsOnNeuron[count] = lineInt[j];
                        count += 1;
                    }
                    if (count == numNeurOnPrevLayer)//This is CORRECT -> if count surpassed the number of neurOnPrevLayer
                    {
                        weightsOnLayer.Add(weightsOnNeuron);
                        weightsOnNeuron = new float[numNeurOnPrevLayer];
                        count = 0;
                    }
                }

                weightsList.Add(weightsOnLayer.ToArray());
            }
            this.SetWeightsWith(weightsList.ToArray());

            //Get Biases Data
            for (int i = noLayers, lay = 0; i < noLayers * 2; i++, lay++)
            {
                string[] lineStr = fileLines[i].Split("b,");//one more read than neccesary

                string str = lay.ToString() + ":";
                float[] lineInt = new float[lineStr.Length - 1];
                Functions.ConvertStrArrToFloatArr(lineStr, ref lineInt);
                this.biases[lay] = lineInt;
            }

            //Get Fitness Data
            string fitStr = fileLines[fileLines.Count - 1];
            fitStr = fitStr.Substring(0, fitStr.Length - 3);
            float fit = float.Parse(fitStr);
            this.SetFitness(fit);
        }
        public static void WriteBrain(in NeuralNetwork net, string path)
        {
            //Json serialization does not support multidimensionalArrays
            /*string json = JsonUtility.ToJson(net);
            path = path.Substring(0, path.Length - 4) + ".json";
            System.IO.File.WriteAllText(path, json);
            return;*/


            StringBuilder data = new StringBuilder();

            //AddLayers
            data.Append(string.Join("n,", net.GetLayers()));
            data.Append("n,\n");

            //AddWeights
            StringBuilder weightsSB = new StringBuilder();
            float[][][] weights = net.GetWeights();
            foreach (float[][] layWeights in weights)
            {
                for (int i = 0; i < layWeights.Length; i++)
                {
                    weightsSB.Append(string.Join("w,", layWeights[i]));
                    weightsSB.Append("w,");
                }
                weightsSB.Append("\n");
            }
            data.Append(weightsSB);

            //AddBiases
            StringBuilder biasesSB = new StringBuilder();
            float[][] biases = net.GetBiases();
            for (int i = 0; i < biases.Length; i++)
            {
                biasesSB.Append(string.Join("b,", biases[i]));
                if (i == 0)
                    biasesSB.Append("b, @input layer biases are never used\n");
                else biasesSB.Append("b,\n");
            }
            data.Append(biasesSB);

            data.Append(net.GetFitness().ToString());
            data.Append("fit\n");
            File.WriteAllText(path, data.ToString());
        }

        public float[] ForwardPropagation(float[] inputs)
        {
            SetInputs(inputs);

            for (int l = 1; l < layers.Length; l++)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    float value = biases[l][n];
                    for (int p = 0; p < neurons[l - 1].Length; p++)
                    {
                        value += weights[l - 1][n][p] * neurons[l - 1][p];
                    }

                    neurons[l][n] = value;

                    if (l != layers.Length - 1)
                        neurons[l][n] = Activate(value, false);
                    else if (outputActivation != ActivationFunctionType.SoftMax)
                        neurons[l][n] = Activate(value, true);



                }
                if (l == layers.Length - 1 && outputActivation == ActivationFunctionType.SoftMax)
                    Functions.ActivationFunctionSoftMax(ref neurons[layers.Length - 1]);
            }

            return neurons[neurons.Length - 1]; //Return the last neurons layer with their values (OUTPUT)
        }

        //-----------------INITIALIZATION-----------------//
        protected void InitializeLayers(int[] layers)
        {
            this.layers = new int[layers.Length];
            SetLayersWith(layers);
        }
        protected void InitializeNeuronsAndBiases(bool emptyBiases)
        {
            List<float[]> neuronsList = new List<float[]>();
            List<float[]> biasesList = new List<float[]>();
            foreach (int neur in layers)
            {
                neuronsList.Add(new float[neur]);
                biasesList.Add(new float[neur]);
            }
            neurons = neuronsList.ToArray();

            if (!emptyBiases)
                for (int i = 1; i < biasesList.Count; i++)
                    Initialize(biasesList[i]);
            biases = biasesList.ToArray();
        }
        protected void InitializeWeights(bool empty)
        {
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> weightsOnLayerList = new List<float[]>();
                int neuronsInPreviousLayer = layers[i - 1];

                for (int n = 0; n < neurons[i].Length; n++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];
                    if (!empty)
                        Initialize(neuronWeights);
                    weightsOnLayerList.Add(neuronWeights);
                }
                weightsList.Add(weightsOnLayerList.ToArray());
            }
            weights = weightsList.ToArray();
        }

        protected void Initialize(float[] axons)
        {

            switch (initialization)
            {
                case InitializationFunctionType.StandardNormal1:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.InitializationFunctionStandardNormal(0.15915f, 2f, 0.3373f);
                    }
                    break;
                case InitializationFunctionType.StandardNormal2:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.InitializationFunctionStandardNormal(0.15915f, 0.2f, 0.3373f);
                    }
                    break;
                case InitializationFunctionType.StandardNormal3:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.InitializationFunctionStandardNormal(0.15915f, 2f, 0.722f);
                    }
                    break;
                case InitializationFunctionType.RandomValue:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = UnityEngine.Random.value;
                    }
                    break;
                default:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.InitializationFunctionStandardNormal(0.15915f, 2f, 0.3373f);
                    }
                    break;

            }
        }

        //--------------------MUTATIONS---------------------//
        public void MutateWeightsAndBiases()
        {
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    for (int k = 0; k < weights[i][j].Length; k++)
                        MutateOneWeightOrBias(ref weights[i][j][k]);

            for (int i = 0; i < biases.Length; i++)
                for (int j = 0; j < biases[i].Length; j++)
                    MutateOneWeightOrBias(ref biases[i][j]);
        }
        protected void MutateOneWeightOrBias(ref float weightOrBias)
        {

            switch (mutation)
            {
                case MutationStrategy.Classic:
                    Functions.ClassicMutation(ref weightOrBias);
                    break;
                case MutationStrategy.LightPercentage:
                    Functions.LightPercentageMutation(ref weightOrBias);
                    break;
                case MutationStrategy.LightValue:
                    Functions.LightValueMutation(ref weightOrBias);
                    break;
                case MutationStrategy.StrongPercentage:
                    Functions.StrongPercentagegMutation(ref weightOrBias);
                    break;
                case MutationStrategy.StrongValue:
                    Functions.StrongValueMutation(ref weightOrBias);
                    break;
                case MutationStrategy.Chaotic:
                    Functions.ChaoticMutation(ref weightOrBias);
                    break;
                default:
                    Functions.ClassicMutation(ref weightOrBias);
                    break;
            }
        }


        //---------------------FITNESS------------------------//
        public int CompareTo(NeuralNetwork other)
        {
            if (other == null) return 1;

            if (this.fitness > other.fitness) return 1;
            if (this.fitness < other.fitness) return -1;
            return 0;
        }
        public void AddFitness(float fit)
        {
            this.fitness += fit;
        }
        public void SetFitness(float fit)
        {
            this.fitness = fit;
        }
        public float GetFitness()
        {
            return this.fitness;
        }

        //--------------------------ACTIVATION---------------------------//
        static float Activate(float value, bool isOutputLayer = false)
        {
            ActivationFunctionType where = isOutputLayer == true ? outputActivation : activation;

            switch (where)
            {
                case ActivationFunctionType.Tanh:
                    return Functions.ActivationFunctionTanh(value);
                case ActivationFunctionType.Sigmoid:
                    return Functions.ActivationFunctionSigmoid(value);
                case ActivationFunctionType.Relu:
                    return Functions.ActivationFunctionReLU(value);
                case ActivationFunctionType.LeakyRelu:
                    return Functions.ActivationFunctionLeakyReLU(value);
                case ActivationFunctionType.BinaryStep:
                    return Functions.ActivationFunctionBinaryStep(value);
                case ActivationFunctionType.Silu:
                    return Functions.ActivationFunctionSiLU(value);
                default:
                    return 0f;
            }
        }
        static float ActivationDerivative(float value, bool isOutputLayer)
        {
            ActivationFunctionType where = isOutputLayer == true ? outputActivation : activation;

            switch (where)
            {
                case ActivationFunctionType.Tanh:
                    return Functions.DerivativeTanh(value);
                case ActivationFunctionType.Sigmoid:
                    return Functions.DerivativeSigmoid(value);
                case ActivationFunctionType.Relu:
                    return Functions.DerivativeReLU(value);
                case ActivationFunctionType.LeakyRelu:
                    return Functions.DerivativeLeakyReLU(value);
                case ActivationFunctionType.BinaryStep:
                    return Functions.DerivativeBinaryStep(value);
                case ActivationFunctionType.Silu:
                    return Functions.DerivativeSiLU(value);
                default:
                    return 0;
            }


        }

        //-------------- GETTERS------------------//
        public int[] GetLayers()
        {
            return layers;
        }
        public float[][][] GetWeights()
        {
            return weights;
        }
        public float[][] GetBiases()
        {
            return biases;
        }
        public int GetInputsNumber()
        {
            return layers[0];
        }
        public int GetOutputsNumber()
        {
            return layers[layers.Length - 1];
        }

        public float GetWeight(int layerAnd1, int neuronIndex, int weightIndex)
        {
            return weights[layerAnd1][neuronIndex][weightIndex];
        }
        public float GetBias(int layerAnd1, int neuron)
        {
            return biases[layerAnd1][neuron];
        }
        public void AddToWeight(int layerAnd1, int neuronIndex, int weightIndex, float value)
        {
            weights[layerAnd1][neuronIndex][weightIndex] += value;
        }
        public void AddToBias(int layerAnd1, int neuron, float value)
        {
            biases[layerAnd1][neuron] += value;
        }
        //------------------SETTERS---------------//
        public void SetLayersWith(int[] layers)
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }
        }
        public void SetWeightsWith(float[][][] weightss)
        {
            for (int i = 0; i < weightss.Length; i++)
                for (int j = 0; j < weightss[i].Length; j++)
                    for (int k = 0; k < weightss[i][j].Length; k++)
                        this.weights[i][j][k] = weightss[i][j][k];
        }
        public void SetBiasesWith(float[][] biasess)
        {
            for (int i = 0; i < biasess.Length; i++)
                for (int j = 0; j < biasess[i].Length; j++)
                    this.biases[i][j] = biasess[i][j];
        }
        //------------------------------------------------------------------------HEURISTIC ------------------------------------------------------------------------------//
        #region HEURISTIC


        private Node[][] nodes;
        float batchError = 0;
        public void Learn(float[] inputs, float[] desiredOuputs, float learnRate, LossFunctionType lossFunction)
        {
            float[] outputs = CalculateOutputs(inputs);
            batchError = CalculateOutputNodesCost(outputs, desiredOuputs, lossFunction);

            BackPropagation(learnRate);


        }
        private void BackPropagation(float learnRate)
        {
            for (int weightLayer = layers.Length - 2; weightLayer >= 0; weightLayer--)//parse each weights layer from END to BEGINING
            {
                float[][] weightsCost = null;
                float[] biasesCost = null;
                CalculateGradientsOfLayer(weightLayer, ref weightsCost, ref biasesCost);//Get the modifications needed for each weight and bias in that layer
                ApplyGradientsOnLayer(weightLayer, weightsCost, biasesCost, learnRate);//Apply the modifications by a specific learnRate
                CalculateNodesCost(weightLayer);//Calculate the valuesInDerivative of the nodes from the left
            }
        }

        private float[] CalculateOutputs(float[] inputs)
        {
            ///normal forward propagation
            /// + calculates valueIn and valueOut of nodes
            SetInputs(inputs);
            nodes = new Node[layers.Length][];

            nodes[0] = new Node[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
                nodes[0][i].valueOut = inputs[i];


            for (int l = 1; l < layers.Length; l++)
            {
                nodes[l] = new Node[neurons[l].Length];

                for (int n = 0; n < neurons[l].Length; n++)
                {

                    float value = biases[l][n];
                    int previousLayerNeuronsNumber = layers[l - 1];
                    for (int p = 0; p < previousLayerNeuronsNumber; p++)
                    {
                        value += weights[l - 1][n][p] * neurons[l - 1][p];
                    }

                    nodes[l][n].valueIn = value;
                    neurons[l][n] = value;//is activated after

                    if (l < layers.Length - 1)
                        neurons[l][n] = Activate(value, false);
                    else if (outputActivation != ActivationFunctionType.SoftMax)
                        neurons[l][n] = Activate(value, true);

                    nodes[l][n].valueOut = neurons[l][n];

                }

                ///SPECIAL CASE
                if (l == layers.Length - 1 && outputActivation == ActivationFunctionType.SoftMax)
                {
                    int neuronsOnLastLayer = layers[l];

                    //Get values In  (it works also for values out because the values are passed normally without Activation =: softMax activation is made after all node values are known)
                    float[] valuesIn = new float[neuronsOnLastLayer];

                    for (int n = 0; n < neurons[l].Length; n++)
                        valuesIn[n] = nodes[l][n].valueIn;

                    //Activate them
                    Functions.ActivationFunctionSoftMax(ref valuesIn);

                    //Set values Out
                    for (int n = 0; n < nodes[l].Length; n++)
                    {
                        neurons[l][n] = valuesIn[n];
                        nodes[l][n].valueOut = neurons[l][n];
                    }


                }

            }
            return neurons[neurons.Length - 1];
        }
        private void CalculateGradientsOfLayer(int weightLayerIndex, ref float[][] wCost, ref float[] bCost)
        {
            //this is a weight layer
            //node layer is always + 1 up
            int outNeurons = layers[weightLayerIndex + 1];
            int inNeurons = layers[weightLayerIndex];

            wCost = new float[outNeurons][];
            bCost = new float[outNeurons];

            for (int i = 0; i < outNeurons; i++)
            {
                wCost[i] = new float[inNeurons];
                for (int j = 0; j < inNeurons; j++)
                {
                    wCost[i][j] = nodes[weightLayerIndex][j].valueOut * nodes[weightLayerIndex + 1][i].costValue;
                }

                bCost[i] = 1 * nodes[weightLayerIndex + 1][i].costValue;
            }
        }
        private void ApplyGradientsOnLayer(int weightLayerIndex, float[][] wCost, float[] bCost, float learnRate)
        {
            int outNeurons = layers[weightLayerIndex + 1];
            int inNeurons = layers[weightLayerIndex];

            for (int i = 0; i < outNeurons; i++)
            {
                for (int j = 0; j < inNeurons; j++)
                {
                    weights[weightLayerIndex][i][j] -= wCost[i][j] * learnRate;
                }
                biases[weightLayerIndex + 1][i] -= bCost[i] * learnRate;
            }
        }

        float CalculateOutputNodesCost(float[] outputs, float[] expectedOutputs, LossFunctionType lossfunc)
        {
            //calculates average error of output nodes
            //calculates output nodes costValue

            if (outputActivation == ActivationFunctionType.SoftMax)
                return CalculateOutputNodesCostForSoftMax(outputs, expectedOutputs, lossfunc);

            float cost = 0;
            for (int i = 0; i < outputs.Length; i++)
            {

                if (lossfunc == LossFunctionType.Quadratic)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.QuadraticNodeCostDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(nodes[nodes.Length - 1][i].valueIn, true);
                    cost += Functions.QuadraticNodeCost(outputs[i], expectedOutputs[i]);
                }
                else if (lossfunc == LossFunctionType.Absolute)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.AbsoluteNodeCostDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(nodes[nodes.Length - 1][i].valueIn, true);
                    cost += Functions.AbsoluteNodeCost(outputs[i], expectedOutputs[i]);
                }
                else if (lossfunc == LossFunctionType.CrossEntropy)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.CrossEntropyNodeCostDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(nodes[nodes.Length - 1][i].valueIn, true);
                    float localCost = Functions.CrossEntropyNodeCost(outputs[i], expectedOutputs[i]);
                    cost += float.IsNaN(localCost) ? 0 : localCost;
                }

            }

            return cost;
        }//LOSS FUNCTION
        float CalculateOutputNodesCostForSoftMax(float[] outputs, float[] expectedOutputs, LossFunctionType lossfunc)
        {
            float cost = 0f;
            float[] derivatedInValues = new float[outputs.Length];
            for (int i = 0; i < derivatedInValues.Length; i++)
                derivatedInValues[i] = nodes[nodes.Length - 1][i].valueIn;
            Functions.DerivativeSoftMax(ref derivatedInValues);

            for (int i = 0; i < outputs.Length; i++)
            {
                if (lossfunc == LossFunctionType.Quadratic)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.QuadraticNodeCostDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                    cost += Functions.QuadraticNodeCost(outputs[i], expectedOutputs[i]);
                }
                else if (lossfunc == LossFunctionType.CrossEntropy)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.CrossEntropyNodeCostDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                    float localCost = Functions.CrossEntropyNodeCost(outputs[i], expectedOutputs[i]);
                    cost += float.IsNaN(localCost) ? 0 : localCost;
                }
                else if (lossfunc == LossFunctionType.Absolute)
                {
                    nodes[nodes.Length - 1][i].costValue = Functions.AbsoluteNodeCostDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                    cost += Functions.AbsoluteNodeCost(outputs[i], expectedOutputs[i]);
                }
            }

            return cost;
        }//LOSS FUNCTION USED WHEN SOFTMAX USED

        void CalculateNodesCost(int layer)
        {
            //IT DOES NOT APPLY FOR OUTPUT NEURON LAYER and INPUT LAYER
            if (layer == 0)
                return;
            int nodesNum = nodes[layer].Length;
            int nextLayerNeuronsNum = layers[layer + 1];

            //The node value is equal to:
            // = Sum(nextLayerNeuron * connectionWeight) * Activation'(nodeValue.valueIN);

            for (int i = 0; i < nodesNum; i++)
            {
                nodes[layer][i].costValue = 0;
                for (int j = 0; j < nextLayerNeuronsNum; j++)
                {
                    nodes[layer][i].costValue += nodes[layer + 1][j].costValue * weights[layer][j][i]; // sum of each nextNeuron*connectionWeight;
                }

                nodes[layer][i].costValue *= ActivationDerivative(nodes[layer][i].valueIn, false);
            }
        }

        public float GetError()
        {
            return batchError;
        }
        //------------------------------------------------------------------------END HEURISTIC ------------------------------------------------------------------------------//
        #endregion
        //--------------COMPLEMENTARY METHODS---------------//
        protected void SetInputs(float[] inputs)
        {
            if (inputs.Length != neurons[0].Length)
                Debug.LogError("<color=red>The number of inputs received is " + inputs.Length + " and the number of input neurons is " + neurons[0].Length + "</color>.<color=grey> Expect some inputs to be ignored!</color>");


            for (int i = 0; i < neurons[0].Length; i++)
            {
                neurons[0][i] = inputs[i];
            }
        }


    }
    public class AgentBase : UnityEngine.MonoBehaviour
    {
        [Header("===== Agent Properties =====")]
        public BehaviorType behavior = BehaviorType.Static;
        [Tooltip("@path of a brain model\n@name of the newly created brain model")] public string path = null;
        [Tooltip("@if has brain: saves current brain data\n@else: creates a brain using Network Properties\n@folder: StreamingAssets/Neural_Networks. \n@default naming format or uses Path")] public bool SaveBrain = false;
        public NeuralNetwork network = null;
        List<PosAndRot> initialPosition = new List<PosAndRot>(); static int parseCounter = 0;

        [Header("===== Network Properties =====")]
        [SerializeField, Range(1, 50), Tooltip("The number of Inputs that the Agent will receive")] private int sensorSize = 2;
        [SerializeField, Range(1, 50), Tooltip("The number of Outputs that the Agent will return")] private int actionSize = 2;
        [SerializeField, Tooltip("Each element is a hidden layer\nEach value is the number of neurons\n@biases not count")] private List<uint> hiddenLayers;

        [Space(20)]
        [EnumMember]
        [Tooltip("@activation function used in hidden layers")]
        public ActivationFunctionType activationType = ActivationFunctionType.Tanh;
        [Tooltip("@activation function used for output layer\n@influences the actionBuffer values")]
        public ActivationFunctionType outputActivationType = ActivationFunctionType.Tanh;
        [Tooltip("@initializes weights and biases of a newly created network")]
        public InitializationFunctionType initializationType = InitializationFunctionType.StandardNormal1;


        [Header("===== Heuristic Properties =====")]
        public HeuristicModule module = HeuristicModule.Append;
        [Tooltip("@do not append/write samples where expected outputs are null\n@expected outputs are considered null if all action vector elements are equal to 0")] public bool killStaticActions = false;
        [Tooltip("@path of the training data file\nname of the newly created training data file")] public string samplesPath = null;
        [Range(0, 300), Tooltip("@data collection time.\n@data_size = sessionLength * avgFPS")] public float sessionLength = 60;
        [Range(1, 1000), Tooltip("@number of parsings through the training batch.")] public uint epochs = 10;
        [Tooltip("@reset the environment transforms when agent action ended")] public GameObject Environment;
        [Tooltip("@watch the progression of the error\n@if is noisy, decrease the learnRate\n@if stagnates, increase the learnRate")] public RectTransform errorGraph;


        [Space(20)]//ADVANCED
        [SerializeField, Tooltip("@read only\n@shows average error of the current epoch\n@sum of all output nodes errors")] private float error;
        [Range(0, 1), Tooltip("@modification strength per epoch")] public float learnRate = 0.01f;
        [Tooltip("@loss function type")] public LossFunctionType costType = LossFunctionType.Quadratic;

        //ONLY HEURISTIC 
        private List<Sample> batch;
        float avgError = 0;
        private List<PosAndRot> environmentInitialTransform;
        //ErrorStatistic
        bool callStatistic = false;
        float maxErrorFound = 0;
        uint startingEpochs = 0;
        List<Vector2> errorPoints = new List<Vector2>();




        protected virtual void Awake()
        {
            GetAllTransforms();
            //On heuristic and self the brains are made directly in CollectHeuristicData and SelfAction,this may cause in action lag

            startingEpochs = epochs;

            //Precaution
            if (activationType == ActivationFunctionType.SoftMax)
            {
                Debug.Log("<color=#4db8ff>SoftMax</color> cannot be an activation function for input or hidden layers. Now is set to <color=#4db8ff>Tanh</color>!");
                activationType = ActivationFunctionType.Tanh;
            }
            if (behavior == BehaviorType.Manual || behavior == BehaviorType.Heuristic)
                HeuristicPreparation();
        }
        protected virtual void Update()
        {
            //SmallChecks
            BUTTONSaveBrain();
            if (behavior == BehaviorType.Self)
                SelfAction();
            else if (behavior == BehaviorType.Manual)
                ManualAction();
            else if (behavior == BehaviorType.Heuristic)
                HeuristicAction();

        }

        void SelfAction()
        {
            if (network == null)
            {
                if (path == null || path == "" || new FileInfo(path).Length == 0)
                {
                    Debug.LogError("<color=red>Cannot Self Control because the Brain Path uploaded is invalid</color>");
                    return;
                }
                this.network = new NeuralNetwork(path);

                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;

            }
            SensorBuffer sensorBuffer = new SensorBuffer(network.GetInputsNumber());
            CollectObservations(ref sensorBuffer);
            ActionBuffer actionBuffer = new ActionBuffer(network.ForwardPropagation(sensorBuffer.GetBuffer()));
            OnActionReceived(in actionBuffer);
        }
        void ManualAction()
        {
            ActionBuffer buffer = new ActionBuffer(this.network.GetOutputsNumber());
            Heuristic(ref buffer);
            OnActionReceived(in buffer);
        }
        void HeuristicAction()
        {
            if (module == HeuristicModule.Learn)
                ProcessHeuristicData();
            else
                CollectHeuristicData();
        }

        //-------------------------------------------HEURISTIC TRAINING--------------------------------------------------//
        void HeuristicPreparation()
        {
            HeuristicEnvironmentSetup();
            HeuristicOnSceneReset();

            if (network == null)
            {
                if (path == null || path == "" || new FileInfo(path).Length == 0)
                {
                    Debug.LogError("<color=red>Brain Path is invalid</color>");
                    return;
                }
                this.network = new NeuralNetwork(path);

                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;

                HeuristicOnSceneReset();
            }

            if (behavior == BehaviorType.Manual)
                return;

            batch = new List<Sample>();
            if (module == HeuristicModule.Learn)
            {
                Debug.Log("<color=#64de18>Collecting data from file <color=grey>" + samplesPath + "</color>...</color>");
                if (samplesPath == null || samplesPath == "" || new FileInfo(samplesPath).Length == 0)
                {
                    Debug.Log("<color=red>Heuristic samples file is invalid.</color>");
                    behavior = BehaviorType.Static;
                    return;
                }


                string[] stringBatch = File.ReadAllLines(samplesPath);
                for (int i = 0; i < stringBatch.Length / 2; i++)
                    batch.Add(GetSampleFromData(stringBatch[i * 2], stringBatch[i * 2 + 1]));


                learnRate /= batch.Count;
                Debug.Log("<color=#64de18>Data arrayed. <color=#e405fc>" + batch.Count + "</color> samples found. The agent started the learning process. Do not stop the simulation.</color>");
            }
            else
            {
                try
                {
                    FileInfo fi = new FileInfo(samplesPath);
                    if (fi.Exists && fi.Length > 0)
                    {
                        Debug.Log("<color=#64de18>Collecting data from user...</color>");
                        return;
                    }
                }
                catch { }



                if (samplesPath == null || samplesPath == "" || samplesPath == " ")
                    samplesPath = GetHeuristicSamplesPath() + "TrainingData_" + UnityEngine.Random.Range(0, 1000).ToString() + ".txt";
                else
                    samplesPath = GetHeuristicSamplesPath() + samplesPath + ".txt";

                Debug.Log("<color=64de18>Training data file  <color=grey>" + samplesPath + "</color> initialized.</color>");
                Debug.Log("<color=#64de18>Collecting data from user...</color>");
            }



        }
        void CollectHeuristicData()
        {

            if (sessionLength <= 0)
            {
                behavior = BehaviorType.Static;
                if (module == HeuristicModule.Append)
                {
                    Debug.Log("<color=#64de18>Appending <color=#e405fc>" + batch.Count + " </color>training samples...</color>");
                    File.AppendAllLines(samplesPath, GetLinesFromBatchList());
                    Debug.Log("<color=#64de18>Data appended succesfully.</color>");
                }
                else if (module == HeuristicModule.Write)
                {
                    Debug.Log("<color=#64de18>Writing <color=#e405fc>" + batch.Count + " </color>training samples...</color>");
                    File.WriteAllLines(samplesPath, GetLinesFromBatchList());
                    Debug.Log("<color=#186ede>Data written succesfully.</color>");
                }
                return;
            }


            sessionLength -= Time.deltaTime;
            //Get inputs
            SensorBuffer inputs = new SensorBuffer(network.GetInputsNumber());
            CollectObservations(ref inputs);
            //Get userOutputs
            ActionBuffer desiredOutputs = new ActionBuffer(network.GetOutputsNumber());
            Heuristic(ref desiredOutputs);
            OnActionReceived(desiredOutputs);

            Sample sample = new Sample();
            sample.inputs = inputs.GetBuffer();
            sample.expectedOutputs = desiredOutputs.GetBuffer();

            //Check if null inputs KILLABLE
            if (killStaticActions)
            {
                foreach (var item in sample.expectedOutputs)
                    if (item != 0)
                    {
                        batch.Add(sample);
                        return;
                    }
            }
            else batch.Add(sample);
        }
        void ProcessHeuristicData()
        {
            if (epochs > 0)
            {
                ProcessHeuristicSample();
                epochs--;
                error = avgError;
                callStatistic = true;
            }
            else
            {
                Debug.Log("<color=#4db8ff>Heuristic training has ended succesfully.</color><color=grey> Watch your agent current performance.</color>");
                NeuralNetwork.WriteBrain(network, path);

                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;


                ResetEnvironmentToInitialPosition();
                ResetToInitialPosition();
                behavior = BehaviorType.Self;
            }
        }
        private void ProcessHeuristicSample()
        {
            avgError = 0;
            foreach (Sample sample in batch)
            {
                network.Learn(sample.inputs, sample.expectedOutputs, learnRate, costType);
                avgError += network.GetError();
            }
            avgError /= batch.Count;
        }

        private void OnDrawGizmos()
        {
            if (!callStatistic)
                return;
            Gizmos.matrix = errorGraph.localToWorldMatrix;
            float xSize = errorGraph.rect.width;
            float ySize = errorGraph.rect.height / 2;

            Vector2 zero = new Vector2(-xSize / 2, 0);
            float zeroX = -xSize / 2;

            //Draw AXIS
            Gizmos.color = Color.white;
            Gizmos.DrawLine(zero, new Vector2(zeroX, ySize));//up
            Gizmos.DrawLine(zero, new Vector2(-zeroX, 0f));//right
            Gizmos.DrawSphere(zero, 5f);
            //Draw Arrows
            float arrowLength = 10f;
            //Y
            Gizmos.DrawLine(new Vector2(zeroX, ySize), new Vector2(zeroX - arrowLength, ySize - arrowLength));
            Gizmos.DrawLine(new Vector2(zeroX, ySize), new Vector2(zeroX + arrowLength, ySize - arrowLength));
            //X
            Gizmos.DrawLine(new Vector2(-zeroX, 0), new Vector2(-zeroX - arrowLength, -arrowLength));
            Gizmos.DrawLine(new Vector2(-zeroX, 0), new Vector2(-zeroX - arrowLength, +arrowLength));
            float xUnit;
            float yUnit;
            if (error > maxErrorFound)
            {
                float oldMaxError = maxErrorFound;
                maxErrorFound = error;
                xUnit = xSize / startingEpochs;
                yUnit = ySize / maxErrorFound;
                for (int i = 0; i < errorPoints.Count; i++)
                {
                    errorPoints[i] = new Vector2(zeroX + xUnit * i, errorPoints[i].y * (oldMaxError / maxErrorFound));
                }
            }
            else
            {
                xUnit = xSize / startingEpochs;
                yUnit = ySize / maxErrorFound;
            }


            Vector2 newErrorPoint = new Vector2(zeroX + xUnit * errorPoints.Count, yUnit * error);
            errorPoints.Add(newErrorPoint);

            //Draw Dots
            Gizmos.color = Color.blue;
            foreach (Vector2 point in errorPoints)
                Gizmos.DrawSphere(point, .5f);

            //Draw Lines
            Gizmos.color = Color.green;
            //Gizmos.DrawLine(zero, errorPoints[0]);
            for (int i = 0; i < errorPoints.Count - 1; i++)
                Gizmos.DrawLine(errorPoints[i], errorPoints[i + 1]);


            callStatistic = false;
        }
        private void HeuristicEnvironmentSetup()
        {
            if (Environment == null)
                return;

            environmentInitialTransform = new List<PosAndRot>();
            GetAllTransforms(Environment.transform, ref environmentInitialTransform);
        }
        private void ResetEnvironmentToInitialPosition()
        {
            if (Environment == null)
                return;
            ApplyAllTransforms(ref Environment, environmentInitialTransform);
        }
        private Sample GetSampleFromData(string inputs, string expectedOuputs)
        {
            Sample newSample = new Sample();

            string[] inp = inputs.Split(',');
            string[] eouts = expectedOuputs.Split(',');

            newSample.expectedOutputs = new float[eouts.Length];
            newSample.inputs = new float[inp.Length];

            ConvertStrArrToFloatArr(inp, ref newSample.inputs);
            ConvertStrArrToFloatArr(eouts, ref newSample.expectedOutputs);
            return newSample;
        }
        private string[] GetLinesFromBatchList()
        {
            string[] lines = new string[batch.Count * 2];
            int i = 0;
            foreach (Sample sample in batch)
            {
                StringBuilder LINE = new StringBuilder();
                foreach (float item in sample.inputs)
                {
                    LINE.Append(item);
                    LINE.Append(",");
                }
                LINE.Length--;
                lines[i++] = LINE.ToString();
                LINE.Clear();
                foreach (float item in sample.expectedOutputs)
                {
                    LINE.Append(item);
                    LINE.Append(",");
                }
                LINE.Length--;
                lines[i++] = LINE.ToString();

            }
            return lines;
        }
        //-------------------------------------------FOR USE BY USER----------------------------------------------//
        /// <summary>
        /// Method to override in order to use Heuristic or Manual mode. Fulfill the ActionBuffer parameter
        /// by your keyboard/mouse inputs with any float values.
        /// <para>Method to use: SetAction(index, value)</para>
        /// </summary>
        /// <param name="actionsOut"></param>
        protected virtual void Heuristic(ref ActionBuffer actionsOut)
        {

        }
        /// <summary>
        /// Method to override to make your agent observe the environment. Fulfill the SensorBuffer with
        /// any values that resembles float type, like Vector3, Transform, Quaternion etc.
        /// <para>Note: The ActionBuffer is a float array. Depending on what kind of observation you append, it can occupy a different space size.
        /// For instance, a Vector3 has 3 floats, so it occupies 3, a Quaternion occupies 4 and so on.</para>
        /// <para>Method to use: AddObservation(Type observation)</para>
        /// </summary>
        /// <param name="sensorBuffer"></param>
        protected virtual void CollectObservations(ref SensorBuffer sensorBuffer)
        {

        }
        /// <summary>
        /// Method to override to make your agent do actions with respect to the observations. Extract each value from the buffer and assign actions for each one.
        /// <para>Methods to use: GetAction(index), GetBuffer(), GetIndexOfMaxValue() -> generally used aside SoftMax</para>
        /// </summary>
        /// <param name="actionBuffer"></param>
        protected virtual void OnActionReceived(in ActionBuffer actionBuffer)
        {

        }
        /// <summary>
        /// General purpose is to move scene objects if needed (getting object references through this script)
        /// <para>Auto called after EndAction() on Heuristic/Manual mode.</para>
        /// </summary>
        protected virtual void HeuristicOnSceneReset()
        {

        }

        /// <summary>
        /// Adds reward to the agent with Self behavior.
        /// <para>Can be added to Static agents when the 'evenIfActionEnded' parameter is true. False by default.</para>
        /// </summary>
        /// <param name="reward"></param>
        /// <param name="evenIfActionEnded">Force rewarding a static agent</param>
        public void AddReward(float reward, bool evenIfActionEnded = false)
        {
            if (behavior == BehaviorType.Manual || behavior == BehaviorType.Heuristic)
                return;
            if (evenIfActionEnded == false && behavior == BehaviorType.Static)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot <color=#18de95>AddReward</color> because neural network is null");
                return;
            }
            network.AddFitness(reward);
        }
        /// <summary>
        /// Sets the reward if the agent with Self behavior.
        /// <para>Can force set to Static agents when the 'evenIfActionEnded' parameter is true.</para>
        /// </summary>
        /// <param name="reward"></param>
        /// <param name="evenIfActionEnded">Force rewarding a static agent. False by default.</param>
        public void SetReward(float reward, bool evenIfActionEnded = false)
        {
            if (behavior == BehaviorType.Manual || behavior == BehaviorType.Heuristic)
                return;
            if (evenIfActionEnded == false && behavior == BehaviorType.Static)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot <color=#18de95>SetReward</color> because neural network is null");
                return;
            }
            network.SetFitness(reward);
        }
        /// <summary>
        /// Sets the behavior to static.
        /// <para>If the behavior was previously Manual or Heuristic, the entire scene resets.</para>
        /// </summary>
        public void EndAction()
        {
            if (behavior == BehaviorType.Self)
                behavior = BehaviorType.Static;
            else if (behavior == BehaviorType.Manual || behavior == BehaviorType.Heuristic)
            {
                ResetToInitialPosition();
                ResetEnvironmentToInitialPosition();
                HeuristicOnSceneReset();
            }
        }

        //--------------------------------------------POSITIONING-----------------------------------------------//
        public void ResetToInitialPosition()
        {
            parseCounter = 1;
            ApplyTransform(initialPosition[0]);
            ApplyChildsTransforms(gameObject, initialPosition);

        } //Only method used externaly

        private void GetAllTransforms()
        {
            parseCounter = 1;
            initialPosition.Add(new PosAndRot(transform.position, transform.localScale, transform.rotation));
            GetChildsTransforms(this.transform);
        }
        private void GetChildsTransforms(UnityEngine.Transform obj)
        {
            foreach (UnityEngine.Transform child in obj.transform)
            {
                PosAndRot tr = new PosAndRot(child.position, child.localScale, child.rotation);
                initialPosition.Add(tr);
                GetChildsTransforms(child);
            }
        }
        static void ApplyChildsTransforms(GameObject obj, in List<PosAndRot> list)
        {
            ///PARSE COUNTER USED SEPARATELY <IT MUST BE INITIALIZED WITH 0></IT>
            for (int i = 0; i < obj.transform.childCount; i++)
            {
                GameObject child = obj.transform.GetChild(i).gameObject;
                ApplyTransformTo(ref child, list[parseCounter]);
                parseCounter++;
                ApplyChildsTransforms(child, list);
            }
        }
        private void ApplyTransform(PosAndRot trnsfrm)
        {
            this.transform.position = trnsfrm.position;
            this.transform.localScale = trnsfrm.scale;
            this.transform.rotation = trnsfrm.rotation;
        }
        static private void ApplyTransformTo(ref GameObject obj, in PosAndRot trnsfrm)
        {
            obj.transform.position = trnsfrm.position;
            obj.transform.localScale = trnsfrm.scale;
            obj.transform.rotation = trnsfrm.rotation;
        }

        #region ENVIRONMENT POSITIONING
        public void GetAllTransforms(UnityEngine.Transform obj, ref List<PosAndRot> inList)
        {
            parseCounter = 1;
            inList.Add(new PosAndRot(obj.position, obj.localScale, obj.rotation));
            GetChildsTransforms(ref inList, obj);
        }
        public void ApplyAllTransforms(ref GameObject obj, in List<PosAndRot> fromList)
        {
            parseCounter = 1;
            ApplyTransform(ref obj, fromList[0]);
            AddChildsInitialTransform(ref obj, fromList);
        }

        public void GetChildsTransforms(ref List<PosAndRot> list, UnityEngine.Transform obj)
        {
            foreach (UnityEngine.Transform child in obj)
            {
                PosAndRot tr = new PosAndRot(child.position, child.localScale, child.rotation);
                list.Add(new PosAndRot(child.position, child.localScale, child.rotation));
                GetChildsTransforms(ref list, child);
            }
        }
        public void AddChildsInitialTransform(ref GameObject obj, List<PosAndRot> list)
        {
            ///PARSE COUNTER USED SEPARATELY <IT MUST BE INITIALIZED WITH 0></IT>
            for (int i = 0; i < obj.transform.childCount; i++)
            {
                GameObject child = obj.transform.GetChild(i).gameObject;
                ApplyTransform(ref child, list[parseCounter]);
                parseCounter++;
                AddChildsInitialTransform(ref child, list);
            }
        }
        public void ApplyTransform(ref GameObject obj, PosAndRot trnsfrm)
        {
            obj.transform.position = trnsfrm.position;
            obj.transform.localScale = trnsfrm.scale;
            obj.transform.rotation = trnsfrm.rotation;
        }
        #endregion


        //--------------------------------------SETTERS AND GETTERS--------------------------------------------//
        public void ForcedSetFitnessTo(float value)
        {
            this.network.SetFitness(0);
        }
        public float GetFitness()
        {
            if (network != null)
                return network.GetFitness();
            else return 0f;
        }
        public string GetPathWithName(string specificName = null)
        {
            StringBuilder pathsb = new StringBuilder();
            pathsb.Append(Application.streamingAssetsPath);
            pathsb.Append("/Neural_Networks/");

            if (!Directory.Exists(pathsb.ToString()))
                Directory.CreateDirectory(pathsb.ToString());
            if (specificName == null)
            {
                pathsb.Append("NetID");
                pathsb.Append(((int)this.gameObject.GetInstanceID()) * (-1));
                pathsb.Append(".txt");
            }
            else
            {
                pathsb.Append(specificName);
                pathsb.Append(".txt");
            }


            return pathsb.ToString();
        }
        private string GetHeuristicSamplesPath()
        {
            string path = Application.streamingAssetsPath;
            path += "/Heuristic_Samples/";
            if (!Directory.Exists(path))
                Directory.CreateDirectory(path);
            return path;
        }
        public List<PosAndRot> GetInitialPosition()
        {
            return initialPosition;
        }

        //---------------------------------------------OPTIONAL---------------------------------------------------//
        void BUTTONSaveBrain()
        {
            if (SaveBrain == false)
                return;

            SaveBrain = false;
            if (network != null)
                NeuralNetwork.WriteBrain(network, GetPathWithName());
            else
            {
                //CreateBrain and Write it
                List<int> lay = new List<int>();
                lay.Add(sensorSize);

                if (hiddenLayers != null)
                    foreach (int neuronsNumber in hiddenLayers)
                    {
                        lay.Add(neuronsNumber);
                    }

                lay.Add(actionSize);
                this.network = new NeuralNetwork(lay.ToArray());
                NeuralNetwork.WriteBrain(network, GetPathWithName(path == null || path == " " || path == "" ? null : path));
            }
        }
        private void ConvertStrArrToFloatArr(string[] str, ref float[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                try
                {
                    arr[i] = float.Parse(str[i]);
                }
                catch (System.Exception e)
                {
                    Debug.LogError(str[i] + " : " + e);
                }

            }
        }
    }
    public class TrainerBase : UnityEngine.MonoBehaviour
    {
        [Header("===== Models =====")]
        [Tooltip("Agent model gameObject used as the ai")] public GameObject AIModel;
        [Tooltip("Brain model used to start the training with")] public string brainModelPath;
        [Tooltip("@resets the dynamic environmental object's positions")] public TrainingType interactionType = TrainingType.NotSpecified;

        [Space(20)]
        [Tooltip("The model used updates in the first next generation\n@tip: use a copy of the brain")] public bool resetBrainModelFitness = false;
        [Tooltip("@save networks of best Ai's before moving to the next generation.\n@number of saves = cbrt(Team Size).\n@folder: /Saves/.\n@last file saved is the best AI")] public bool saveBrains = false;

        [Header("===== Statistics Display =====")]
        [Tooltip("First ObjectOfType<Camera>")] public bool cameraFollowsBestAI = true; GameObject cam; bool isOrtographic; Vector3 perspectiveOffset;
        [Tooltip("Load a Canvas TMPro to watch the current performance of AI's")] public TMPro.TMP_Text Labels = null;
        [Tooltip("Load a Canvas RectTransform to watch a Gizmos graph in SceneEditor")] public RectTransform Graph = null;
        List<float> bestResults;//memorize best results for every episode
        List<float> averageResults;//memorize avg results for every episode

        [Space, Header("===== Training Settings =====")]
        [Range(3, 1000)] public int teamSize = 10;//IT cannot be 1 or 2, otherwise strategies will not work (if there are not 3, strategy 2 causes trouble)
        [Range(1, 10), Tooltip("Episodes needed to run until passing to the next Generation\n@TIP: divide the reward given by this number")] public int episodesPerEvolution = 1;
        [Range(1, 1000), Tooltip("Total Episodes in this Training Session")] public int maxEpisodes = 100; private int currentEpisode = 1;
        [Range(1, 1000), Tooltip("Maximum time allowed per Episode")] public float maxTimePerEpisode = 25f; float timeLeft;

        [Space, Header("===== Strategies =====")]
        [Tooltip("@in the beggining use Strategy1.\n@if AI's performance decreases, switch to Strategy2.\n@finetune the final Brain using Strategy3.")]
        public TrainingStrategy trainingStrategy = TrainingStrategy.Strategy1;
        [Tooltip("@mutates the weights and biases following certain rules")]
        public MutationStrategy mutationStrategy = MutationStrategy.Classic;


        private NeuralNetwork modelNet;
        protected AI[] team;
        private GameObject[] Environments;

        private int currentEnvironment = 0;
        protected List<PosAndRot>[] environmentsInitialTransform;//Every item is the position of a single environment. The item is a list with positions of all environment items
        protected List<PosAndRot>[] agentsInitialTransform;//Every item is the position for a single environment. The item is the AI's initial position for environment i

        int parseCounter = 0;
        bool startTraining = true;


        protected virtual void Awake()
        {
            CreateDir();
            timeLeft = maxTimePerEpisode;
            bestResults = new List<float>();
            averageResults = new List<float>();


            //Cam related
            cam = FindObjectOfType<Camera>().gameObject;
            if (cam.GetComponent<Camera>().orthographic == true)
                isOrtographic = true;
            else
            { isOrtographic = false; perspectiveOffset = cam.transform.position - AIModel.transform.position; }
        }
        protected virtual void Start()
        {
            if (!TrainingPreparation())
            {
                startTraining = false;
                return;
            }
            EnvironmentSetup();
            SetupTeam();

        }
        protected virtual void Update()
        {
            NeuralNetwork.mutation = mutationStrategy;
            if (startTraining)
                Train();
        }
        protected virtual void LateUpdate()
        {
            if (cameraFollowsBestAI)
            {
                if (isOrtographic)
                    OrtographicCameraFollowsBestAI();
                else PerspectiveCameraFollowsBestAI();
            }
        }
        //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------//
        //------------------------------------------TRAINING SETUP-----------------------------------------//
        void CreateDir()
        {
            if (!Directory.Exists(Application.streamingAssetsPath + "/Neural_Networks/"))
                Directory.CreateDirectory(Application.streamingAssetsPath + "/Neural_Networks/");
        }
        bool TrainingPreparation()
        {
            if (AIModel == null)
            {
                Debug.LogError("The training cannot start! Reason: <color=#f27602>No AI Model uploaded</color>");
                return false;
            }
            if (AIModel.GetComponent<Agent>() == null)
            {
                Debug.LogError("The training cannot start! Reason:  <color=#f27602>AI Model is not a Agent</color>");
                return false;
            }
            if (brainModelPath == null || brainModelPath == "")
            {
                Debug.LogError("The training cannot start! Reason:  <color=#f27602>Brain Model not uploaded</color>");
                return false;
            }
            if (!File.Exists(brainModelPath))
            {
                Debug.LogError("The training cannot start! Reason:  <color=#f27602>Brain Model Path uploaded doesn't exists</color>");
                return false;
            }
            modelNet = new NeuralNetwork(brainModelPath);
            if (resetBrainModelFitness)
                modelNet.SetFitness(0f);
            return true;
        }
        /// <summary>
        /// Requires base.SetupTeam() to be called in the beggining.
        /// <para>Can be overridden for pre-training setup, like coloring the agents differently.
        /// </para>
        /// <para>They are auto colored differently if they have a SpriteRenderer component.</para>
        /// </summary>
        protected virtual void SetupTeam()
        {
            //Instatiate AI
            team = new AI[teamSize];
            AIModel.GetComponent<Agent>().behavior = BehaviorType.Static;
            if (interactionType == TrainingType.OneAgentPerEnvironment)
                for (int i = 0; i < Environments.Length; i++)
                {
                    Agent ag = (Agent)Environments[i].transform.GetComponentInChildren(typeof(Agent), true);
                    team[i].agent = ag.gameObject;
                    team[i].agent.SetActive(true);
                    team[i].script = ag;
                    team[i].fitness = 0f;
                }
            else
                for (int i = 0; i < team.Length; i++)
                {
                    GameObject member = Instantiate(AIModel, AIModel.transform.position, AIModel.transform.rotation);
                    team[i].agent = member;
                    team[i].agent.SetActive(true);
                    team[i].script = member.GetComponent<Agent>() as Agent;
                    team[i].fitness = 0f;
                }

            NeuralNetwork.activation = team[0].script.activationType;
            NeuralNetwork.outputActivation = team[0].script.outputActivationType;
            NeuralNetwork.initialization = team[0].script.initializationType;

            //Initialize AI
            for (int i = 0; i < team.Length; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(brainModelPath);
                script.ForcedSetFitnessTo(0f);

                //Mutate Half Of them in the beggining
                if (i % 2 == 0)
                    script.network.MutateWeightsAndBiases();

                script.behavior = BehaviorType.Self;

            }


            ResetAgentsTransform();

            //Colorize AI's if possible
            foreach (var item in team)
            {
                if (item.agent.TryGetComponent<SpriteRenderer>(out var spriteRenderer))
                {
                    if (spriteRenderer == null)
                        break;
                    spriteRenderer.color = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value);
                }
            }

            //Turn Off the model
            AIModel.SetActive(false);
            UpdateDisplay();
        }

        //--------------------------------------------TRAINING PROCESS--------------------------------------//
        void Train()
        {
            timeLeft -= Time.deltaTime;
            EnvironmentAction();

            UpdateFitnessInArray();

            if (AreAllDead() || timeLeft <= 0)
                ResetEpisode();

            if (currentEpisode >= maxEpisodes)
            {
                SaveBrains();
                Debug.Log("<color=#027af2>Training Session Ended!</color>");
                foreach (var item in team)
                    item.script.behavior = BehaviorType.Static;
                startTraining = false;
            }
            UpdateDisplay();
        }
        /// <summary>
        /// Adds actions to the environment [objects] referenced to this script. Use Time.deltaTime if needed.
        /// </summary>
        protected virtual void EnvironmentAction()
        {

        }
        bool AreAllDead()
        {
            foreach (AI item in team)
                if (item.script.behavior == BehaviorType.Self)
                    return false;
            return true;
        }
        protected void ResetEpisode()
        {
            for (int i = 0; i < team.Length; i++)
                OnEpisodeEnd(ref team[i]);//it makes sense to be here

            UpdateFitnessInArray();
            SortTeam();

            timeLeft = maxTimePerEpisode;
            //Next Gen
            if (currentEpisode % episodesPerEvolution == 0)
            {
                //Graph Related
                bestResults.Add(team[team.Length - 1].fitness);
                averageResults.Add(FindAverageResult());

                if (saveBrains == true)
                    SaveBrains();
                switch (trainingStrategy)
                {
                    case TrainingStrategy.Strategy1:
                        NextGenStrategy1();
                        break;
                    case TrainingStrategy.Strategy2:
                        NextGenStrategy2();
                        break;
                    case TrainingStrategy.Strategy3:
                        NextGenStrategy3();
                        break;
                    default:
                        Debug.LogError("Training Strategy is NULL");
                        break;

                }
                ResetFitEverywhere();
            }
            if (interactionType != TrainingType.NotSpecified)
                NextEnvironment();
            ResetEnvironmentTransform();
            ResetAgentsTransform();



            //From static, move to self
            foreach (var item in team)
                item.script.behavior = BehaviorType.Self;

            currentEpisode++;
            OnEpisodeBegin(ref Environments);
        }
        /// <summary>
        /// Adds actions after episode restting. Use-cases: flags activations, environment repositioning etc.
        /// <para>To modify one object in all environments, the parameter array must be parsed. Then, in each environment, search for the object needed and modify it.</para>
        /// <para>For monoenvironment, you can get reference to the objects through the Trainer script and modify their behavior here, without using the parameter.</para>
        /// </summary>
        /// <param name="Environments">array containing each Environment gameObject</param>
        protected virtual void OnEpisodeBegin(ref GameObject[] Environments)
        {

        }
        /// <summary>
        /// Actions before episode resetting. Usually used for post-action rewards, when the agents become static.
        /// <para>This method is called for each AI separately, having a referenced AI parameter.
        /// </para>
        /// <para>AI parameter has 3 different fields: agent, script and fitness. All are described by hovering over them.</para>
        /// </summary>
        /// <param name="ai"></param>
        protected virtual void OnEpisodeEnd(ref AI ai)
        {
            ///Is called at the beggining of ResetEpisode()
        }
        //----------------------------------------------ENVIRONMENTAL----------------------------------------//
        private void EnvironmentSetup()
        {
            // This function will search for a Start object that has the same tag as the AI.
            // If the AI has a Default tag, the first object in the environment will be considered Start and this will be a bug. **Always TAG AI's**
            if (interactionType == TrainingType.NotSpecified)
                return;

            try
            {
                Environments = GameObject.FindGameObjectsWithTag("Environment");
            }
            catch { Debug.LogError("Environment tag doesn't exist. Please create a tag called Environment and assign to an environment!"); interactionType = TrainingType.NotSpecified; return; }


            environmentsInitialTransform = new List<PosAndRot>[Environments.Length];
            agentsInitialTransform = new List<PosAndRot>[Environments.Length];
            for (int i = 0; i < Environments.Length; i++)
            {
                environmentsInitialTransform[i] = new List<PosAndRot>();
                agentsInitialTransform[i] = new List<PosAndRot>();
            }

            if (Environments.Length == 0)
            {
                Debug.Log("There is no Environment found. Make sure your environments have Environment tag.");
                return;
            }
            if (Environments.Length == 1)
            {

                //GetEnvironment transform
                GetAllTransforms(Environments[0].transform, ref environmentsInitialTransform[0]);

                //GetStart transform
                UnityEngine.Transform Start = null;
                foreach (Transform child in Environments[0].transform)
                {
                    if (child.GetComponent<Agent>() == true)
                    { Start = child; break; }
                }
                if (Start == null)//If the monoenvironment doesn't have a start, take as start the AIModel
                    GetAllTransforms(AIModel.transform, ref agentsInitialTransform[0]);
                else
                {
                    GetAllTransforms(Start, ref agentsInitialTransform[0]);
                    Start.gameObject.SetActive(false);
                }

            }
            else
            {
                for (int i = 0; i < Environments.Length; i++)
                {
                    GetAllTransforms(Environments[i].transform, ref environmentsInitialTransform[i]);

                    UnityEngine.Transform StartTransform = null;

                    foreach (Transform child in Environments[i].transform)
                    {
                        if (child.GetComponent<Agent>() == true)
                        { StartTransform = child; break; }
                    }

                    if (StartTransform == null)
                    {
                        Debug.Log("<color=red><color=#a9fc03>" + Environments[i].name + "</color> does not have an agent model inside.</color>");
                        return;
                    }

                    GetAllTransforms(StartTransform, ref agentsInitialTransform[i]);
                    StartTransform.gameObject.SetActive(false);
                }
                if (interactionType == TrainingType.MoreAgentsPerEnvironment)
                    episodesPerEvolution *= Environments.Length;
            }

            if (interactionType == TrainingType.OneAgentPerEnvironment)
                teamSize = Environments.Length;

        }
        private void NextEnvironment()
        {
            currentEnvironment++;
            if (currentEnvironment == Environments.Length)
                currentEnvironment = 0;
        }
        private void ResetEnvironmentTransform()
        {
            if (interactionType == TrainingType.NotSpecified)
                return;
            //SingleLayer Environment - reset only current env
            if (interactionType == TrainingType.MoreAgentsPerEnvironment)
                ApplyAllTransforms(ref Environments[currentEnvironment], in environmentsInitialTransform[currentEnvironment]);
            //MultiLayer Environment - reset all environments
            else
                for (int i = 0; i < Environments.Length; i++)
                    ApplyAllTransforms(ref Environments[i], in environmentsInitialTransform[i]);
        }
        private void ResetAgentsTransform()
        {
            if (interactionType == TrainingType.NotSpecified)
                for (int i = 0; i < team.Length; i++)
                    team[i].script.ResetToInitialPosition();
            else if (interactionType == TrainingType.MoreAgentsPerEnvironment)
                for (int i = 0; i < team.Length; i++)
                    ApplyAllTransforms(ref team[i].agent, in agentsInitialTransform[currentEnvironment]);
            else
                for (int i = 0; i < team.Length; i++)
                    ApplyAllTransforms(ref team[i].agent, in agentsInitialTransform[i]);
        }
        ///---------POSITIONING---------//
        public void GetAllTransforms(UnityEngine.Transform obj, ref List<PosAndRot> inList)
        {
            parseCounter = 1;
            inList.Add(new PosAndRot(obj.position, obj.localScale, obj.rotation));
            GetChildsTransforms(ref inList, obj);
        }
        public void ApplyAllTransforms(ref GameObject obj, in List<PosAndRot> fromList)
        {
            parseCounter = 1;
            ApplyTransform(ref obj, fromList[0]);
            AddChildsInitialTransform(ref obj, in fromList);
        }

        public void GetChildsTransforms(ref List<PosAndRot> list, UnityEngine.Transform obj)
        {
            foreach (UnityEngine.Transform child in obj)
            {
                PosAndRot tr = new PosAndRot(child.position, child.localScale, child.rotation);
                list.Add(new PosAndRot(child.position, child.localScale, child.rotation));
                GetChildsTransforms(ref list, child);
            }
        }
        public void AddChildsInitialTransform(ref GameObject obj, in List<PosAndRot> list)
        {
            ///PARSE COUNTER USED SEPARATELY <IT MUST BE INITIALIZED WITH 0></IT>
            for (int i = 0; i < obj.transform.childCount; i++)
            {
                GameObject child = obj.transform.GetChild(i).gameObject;
                ApplyTransform(ref child, list[parseCounter]);
                parseCounter++;
                AddChildsInitialTransform(ref child, list);
            }
        }
        public void ApplyTransform(ref GameObject obj, PosAndRot trnsfrm)
        {
            obj.transform.position = trnsfrm.position;
            obj.transform.localScale = trnsfrm.scale;
            obj.transform.rotation = trnsfrm.rotation;
        }
        //-----------------------------------------------STATISTICS---------------------------------------------//
        void OrtographicCameraFollowsBestAI()
        {
            Vector3 cameraPosition = cam.transform.position;
            Vector3 targetPosition = team[team.Length - 1].agent.transform.position;
            Vector3 desiredPosition = new Vector3(targetPosition.x, targetPosition.y, cameraPosition.z);

            float smoothness = 0.125f;
            Vector3 smoothPosition = Vector3.Lerp(cameraPosition, desiredPosition, smoothness);

            cam.transform.position = smoothPosition;
        }
        void PerspectiveCameraFollowsBestAI()
        {
            Vector3 cameraPosition = cam.transform.position;
            Vector3 targetPosition = team[team.Length - 1].agent.transform.position;
            Vector3 desiredPosition = targetPosition + perspectiveOffset;

            float smoothness = 0.0625f;
            Vector3 smoothPosition = Vector3.Lerp(cameraPosition, desiredPosition, smoothness);

            cam.transform.position = smoothPosition;
        }
        void UpdateDisplay()
        {
            //Update is called after every EpisodeReset
            if (Labels == null)
                return;

            SortTeam();
            string statColor;
            StringBuilder statData = new StringBuilder();

            {
                Color color = Color.Lerp(Color.green, Color.red, currentEpisode / maxEpisodes);
                Color32 color32 = new Color32();
                ColorConvertor.ConvertColorToColor32(color, ref color32);
                statColor = ColorConvertor.GetRichTextColorFromColor32(color32);
            }
            statData.Append("<b>|Episode: <color=" + statColor + ">");
            statData.Append(currentEpisode);
            statData.Append("</color>\n");

            statData.Append("<b>|Generation: ");
            statData.Append((currentEpisode - 1) / episodesPerEvolution);
            statData.Append("\n");
            {//Colorize
                Color tlcolor = Color.Lerp(Color.red, Color.green, timeLeft / maxTimePerEpisode);
                Color32 tlcolor32 = new Color32();
                ColorConvertor.ConvertColorToColor32(tlcolor, ref tlcolor32);
                statColor = ColorConvertor.GetRichTextColorFromColor32(tlcolor32);
            }
            statData.Append("<b>|Timeleft: <color=" + statColor + ">");
            statData.Append(timeLeft.ToString("0.000"));
            statData.Append("</color>\n");

            statData.Append("|Goal: ");
            statData.Append(modelNet.GetFitness().ToString("0.000"));
            statData.Append("</b>\n\n");
            for (int i = team.Length - 1; i >= 0; --i)
            {
                AI item = team[i];
                StringBuilder line = new StringBuilder();

                //Try COLORIZE
                bool hasColor = true;
                try
                {
                    Color color = item.agent.GetComponent<SpriteRenderer>().color;
                    Color32 color32 = new Color32();
                    ColorConvertor.ConvertColorToColor32(color, ref color32);
                    StringBuilder colorString = new StringBuilder();
                    colorString.Append("#");
                    colorString.Append(ColorConvertor.GetHexFrom(color32.r));
                    colorString.Append(ColorConvertor.GetHexFrom(color32.g));
                    colorString.Append(ColorConvertor.GetHexFrom(color32.b));

                    line.Append("<color=" + colorString.ToString() + ">");
                }
                catch { hasColor = false; }

                line.Append("ID: ");
                line.Append((((int)item.agent.GetInstanceID()) * (-1)).ToString());
                //IF COLORIZED
                if (hasColor)
                    line.Append("</color>");

                line.Append(" | Fitness: ");
                line.Append(item.script.GetFitness().ToString("0.000"));

                if (item.script.behavior == BehaviorType.Self)
                    line.Append(" | <color=green>@</color>");
                else line.Append(" | <color=red>X</color>");
                line.Append("\n");
                statData.AppendLine(line.ToString());
            }
            Labels.text = statData.ToString();
        }
        private void OnDrawGizmos()
        {
            //Draw Graph
            if (Graph == null || modelNet == null)
                return;
            try
            {
                float goal = modelNet.GetFitness();
                Gizmos.matrix = Graph.localToWorldMatrix;
                float xSize = Graph.rect.width;
                float ySize = Graph.rect.height / 2;

                Vector2 zero = new Vector2(-xSize / 2, 0);
                float zeroX = -xSize / 2;

                //Draw AXIS
                Gizmos.color = Color.white;
                Gizmos.DrawLine(zero, new Vector2(zeroX, ySize));//up
                Gizmos.DrawLine(zero, new Vector2(-zeroX, 0f));//right
                Gizmos.DrawSphere(zero, 5f);
                //Draw Arrows
                float arrowLength = 10f;
                //Y
                Gizmos.DrawLine(new Vector2(zeroX, ySize), new Vector2(zeroX - arrowLength, ySize - arrowLength));
                Gizmos.DrawLine(new Vector2(zeroX, ySize), new Vector2(zeroX + arrowLength, ySize - arrowLength));
                //X
                Gizmos.DrawLine(new Vector2(-zeroX, 0), new Vector2(-zeroX - arrowLength, -arrowLength));
                Gizmos.DrawLine(new Vector2(-zeroX, 0), new Vector2(-zeroX - arrowLength, +arrowLength));



                float xUnit = xSize / currentEpisode;
                float yUnit = ySize / goal;

                //Draw Best Dots
                Gizmos.color = Color.yellow;
                List<Vector3> pointsPositions = new List<Vector3>();
                pointsPositions.Add(zero);
                for (int i = 0; i < bestResults.Count; i++)
                {

                    float xPos;
                    float yPos;

                    xPos = zeroX + (i + 1) * xUnit * episodesPerEvolution; //episodesPerEvolution is added, otherwise the graph will remain to short on Xaxis
                    yPos = bestResults[i] * yUnit;   //fitness
                    Vector3 dotPos = new Vector3(xPos, yPos, 0f);
                    pointsPositions.Add(dotPos);
                    Gizmos.DrawSphere(dotPos, 5f);
                }

                //Draw Dots Connection
                Gizmos.color = Color.green;
                for (int i = 0; i < pointsPositions.Count - 1; i++)
                {
                    Gizmos.DrawLine(pointsPositions[i], pointsPositions[i + 1]);
                }



                pointsPositions.Clear();
                //Draw Average Dots
                pointsPositions.Add(zero);
                Gizmos.color = Color.blue;
                for (int i = 0; i < averageResults.Count; i++)
                {
                    float xPos;
                    float yPos;

                    xPos = zeroX + (i + 1) * xUnit * episodesPerEvolution;    //step
                    yPos = averageResults[i] * yUnit;   //fitness
                    Vector3 dotPos = new Vector3(xPos, yPos, 0f);
                    pointsPositions.Add(dotPos);
                    Gizmos.DrawSphere(dotPos, 5f);
                }
                //Draw Dots Connection
                Gizmos.color = Color.grey;
                for (int i = 0; i < pointsPositions.Count - 1; i++)
                {
                    Gizmos.DrawLine(pointsPositions[i], pointsPositions[i + 1]);
                }
            }
            catch { }
            //Draw Neural Network Shape
            try
            {
                float SCALE = .05f;
                Color emptyNeuron = Color.black;
                Color fullNeuron = Color.yellow;
                Color biasColor = Color.green;
                NeuralNetwork nety = team[team.Length - 1].script.network;
                try { emptyNeuron = team[team.Length - 1].agent.GetComponent<SpriteRenderer>().color; } catch { }


                int[] layers = nety.GetLayers();
                float[][][] weights = nety.GetWeights();
                float[][] biases = nety.GetBiases();

                Vector2[][] neuronsPosition = new Vector2[layers.Length][];//starts from up-left
                Vector2[] biasesPosition = new Vector2[layers.Length - 1];//one for each layer

                float xSize = Graph.rect.width;
                float ySize = Graph.rect.height;
                float maxNeuronsInLayers = layers.Max();
                float scale = 1 / (layers.Length * maxNeuronsInLayers) * SCALE;
                float xOffset = -xSize / 2;
                float yOffset = -ySize / 2;

                float layerDistanceUnit = xSize / (layers.Length - 1);
                float neuronDistanceUnit = ySize / (maxNeuronsInLayers) / 2;
                neuronDistanceUnit -= neuronDistanceUnit * 0.15f;//substract 10% to make it a bit smaller - also not substract 1  form maxNeuronsInLayers beacause is one more bias

                //FIND POSITIONS
                for (int layerNum = 0; layerNum < layers.Length; layerNum++)//take each layer individually
                {
                    //float layerYstartPose = (maxNeuronsInLayers - layers[layerNum]) / 2 * neuronDistanceUnit;
                    float layerYStartPose = -(maxNeuronsInLayers - layers[layerNum]) / 2 * neuronDistanceUnit - 50f;//substract 30f to not interact with the graph
                    neuronsPosition[layerNum] = new Vector2[layers[layerNum]];
                    for (int neuronNum = 0; neuronNum < layers[layerNum]; neuronNum++)
                        neuronsPosition[layerNum][neuronNum] = new Vector2(layerNum * layerDistanceUnit + xOffset, layerYStartPose - neuronNum * neuronDistanceUnit);
                    if (layerNum < layers.Length - 1)
                        biasesPosition[layerNum] = new Vector2(layerNum * layerDistanceUnit + xOffset, layerYStartPose - layers[layerNum] * neuronDistanceUnit);
                }

                //Draw biases weights with their normal values
                for (int i = 1; i < neuronsPosition.Length; i++)
                {
                    for (int j = 0; j < neuronsPosition[i].Length; j++)
                    {
                        float weightValue = biases[i][j];
                        if (weightValue > 0)
                            Gizmos.color = Color.blue;
                        else Gizmos.color = Color.red;
                        Gizmos.DrawLine(biasesPosition[i - 1], neuronsPosition[i][j]);
                    }
                }

                //Draw empty weights with their normal values 
                for (int i = 1; i < neuronsPosition.Length; i++)//start from the second layer** keep in mind
                    for (int j = 0; j < neuronsPosition[i].Length; j++)
                        for (int backNeuron = 0; backNeuron < neuronsPosition[i - 1].Length; backNeuron++)
                        {
                            float weightValue = weights[i - 1][j][backNeuron];
                            if (weightValue > 0)
                                Gizmos.color = Color.blue;
                            else
                                Gizmos.color = Color.red;
                            Gizmos.DrawLine(neuronsPosition[i][j], neuronsPosition[i - 1][backNeuron]);

                        }

                //Draw Neurons
                Gizmos.color = emptyNeuron;
                for (int i = 0; i < neuronsPosition.Length; i++)
                    for (int j = 0; j < neuronsPosition[i].Length; j++)
                        Gizmos.DrawSphere(neuronsPosition[i][j], scale * 4000f);

                //Draw Biases
                Gizmos.color = biasColor;
                for (int i = 0; i < biasesPosition.Length; i++)
                {
                    Gizmos.DrawSphere(biasesPosition[i], scale * 4000f);
                }


            }
            catch { }
        }
        float FindAverageResult()
        {
            float result = 0f;
            foreach (AI aI in team)
            {
                result += aI.fitness;
            }
            return result / team.Length;
        }
        //--------------------------------------------TRAINING STRATEGY----------------------------------//
        void NextGenStrategy1()
        {
            /// <summary>
            /// Half worst AI's are replaced with the a single copy of half best AI's, only the copy is mutated
            /// </summary>
            SortTeam();
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentEpisode);
            statistic.Append(" TEAM: <color=#4db8ff>");
            for (int i = team.Length - 1; i >= 0; i--)
            {
                if (i == team.Length / 2 - 1)
                    statistic.Append(" |</color><color=red>");

                statistic.Append(" | ");
                statistic.Append(team[i].fitness);
            }
            statistic.Append(" |</color>");
            float thisGenerationBestFitness = team[team.Length - 1].fitness;
            if (thisGenerationBestFitness < this.modelNet.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.modelNet.GetFitness());
                //update ModelBrain
                NeuralNetwork.WriteBrain(in team[team.Length - 1].script.network, brainModelPath);
                modelNet = new NeuralNetwork(brainModelPath);
            }
            Debug.Log(statistic.ToString());


            //BUILD NEXT GENERATION
            int halfCount = team.Length / 2;
            if (team.Length % 2 == 0)//If Even team Size
                for (int i = 0; i < halfCount; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + halfCount].script.network);
                    script.network.MutateWeightsAndBiases();
                }
            else
                for (int i = 0; i <= halfCount; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + halfCount].script.network);
                    script.network.MutateWeightsAndBiases();
                }
        }
        void NextGenStrategy2()
        {
            /// <summary>
            /// 1/3 of the AI's (the worst) receive best brain and get mutated, for the rest 2/3 the first strategy applies
            /// </summary>
            SortTeam();
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentEpisode);
            statistic.Append(" TEAM: <color=#4db8ff>");

            int somevar = team.Length % 3;
            if (somevar != 0)
                somevar = 1;
            for (int i = team.Length - 1; i >= 0; i--)
            {
                if (i == team.Length - team.Length / 3 - somevar - 1)
                    statistic.Append(" |</color><color=red>");
                else if (i == team.Length / 3 - 1)
                    statistic.Append(" |</color><color=#90a2a2>");
                statistic.Append(" | ");
                statistic.Append(team[i].fitness);
            }
            statistic.Append(" |</color>");
            float thisGenerationBestFitness = team[team.Length - 1].fitness;
            if (thisGenerationBestFitness < this.modelNet.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.modelNet.GetFitness());
                //update ModelBrain
                NeuralNetwork.WriteBrain(in team[team.Length - 1].script.network, brainModelPath);
                modelNet = new NeuralNetwork(brainModelPath);
            }
            Debug.Log(statistic.ToString());

            for (int i = 0; i <= team.Length / 3; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(modelNet);
                script.network.MutateWeightsAndBiases();
            }
            int mod = team.Length % 3;
            if (mod == 0)
                for (int i = team.Length / 3; i < team.Length / 3 * 2; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + team.Length / 3].script.network);
                }
            else if (mod == 1)
                for (int i = team.Length / 3; i <= team.Length / 3 * 2; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + team.Length / 3].script.network);
                }
            else if (mod == 2)
                for (int i = team.Length / 3; i <= team.Length / 3 * 2; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + team.Length / 3 + 1].script.network);
                }
        }
        void NextGenStrategy3()
        {
            /// <summary>
            /// Best AI is reproduced, all of his clones are mutated
            /// </summary>
            SortTeam();
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentEpisode);

            //Place best AI in yellow
            statistic.Append(" TEAM: <color=#e6e600>");
            statistic.Append(" | ");
            statistic.Append(team[team.Length - 1].fitness);

            //Add rest of Ai's with grey
            statistic.Append(" |</color><color=#4db8ff>");
            for (int i = team.Length - 2; i >= 0; i--)
            {
                statistic.Append(" | ");
                statistic.Append(team[i].fitness);
            }
            statistic.Append(" |</color>");

            float thisGenerationBestFitness = team[team.Length - 1].fitness;
            if (thisGenerationBestFitness < this.modelNet.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.modelNet.GetFitness());
                //update ModelBrain
                NeuralNetwork.WriteBrain(in team[team.Length - 1].script.network, brainModelPath);
                modelNet = new NeuralNetwork(brainModelPath);
            }
            Debug.Log(statistic.ToString());

            NeuralNetwork bestAINet = team[team.Length - 1].script.network;
            for (int i = 0; i < team.Length - 1; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(bestAINet);
                script.network.MutateWeightsAndBiases();
            }
        }
        //-------------------------------------------------SORTING------------------------------------//
        void SortTeam()
        {//InsertionSort
            for (int i = 1; i < team.Length; i++)
            {
                var key = team[i];
                int j = i - 1;
                while (j >= 0 && team[j].fitness > key.fitness)
                {
                    team[j + 1] = team[j];
                    j--;
                }
                team[j + 1] = key;
            }
        }
        AI[] SortTeamByQuicksort(ref AI[] arr)
        {
            return null;
            //STACK OVERFLOW PROBLEMS
            /*  return null;
              List<AI> less = new List<AI>();
              List<AI> equal = new List<AI>();
              List<AI> greater = new List<AI>();
              if (arr.Length > 1)
              {
                  float pivot = team[0].fitness;
                  foreach (var item in arr)
                  {
                      if (item.fitness < pivot)
                          less.Add(item);
                      else if (item.fitness == pivot)
                          equal.Add(item);
                      else if (item.fitness > pivot)
                          greater.Add(item);
                  }
                  AI[] lessArr = less.ToArray();
                  AI[] equalArr = equal.ToArray();
                  AI[] greaterArr = greater.ToArray();

                  AI[] newArr = new AI[lessArr.Length + equalArr.Length + greaterArr.Length];
                  SortTeamByQuicksort(ref lessArr).CopyTo(newArr, 0);
                  equalArr.CopyTo(newArr, lessArr.Length);
                  SortTeamByQuicksort(ref greaterArr).CopyTo(newArr, equalArr.Length);
                  return newArr;
              }
              else return arr;*/
        }

        //------------------------------------------COMPLEMENTARY METHODS-----------------------------------// 
        private void ResetFitEverywhere()
        {
            for (int i = 0; i < team.Length; i++)
            {
                team[i].script.ForcedSetFitnessTo(0f);
                team[i].fitness = 0f;
            }
        }
        private void UpdateFitnessInArray()
        {
            //Update fitness in team probArr
            for (int i = 0; i < team.Length; i++)
                team[i].fitness = team[i].script.GetFitness();
        }
        //-------------------------------------------------BUTTONS------------------------------------------//
        void SaveBrains()
        {
            saveBrains = false;

            //mainDir is the main Saves directory
            //saveDir is the directory made for this specific save, it is included in the main Saves directory
            string mainDir = Application.streamingAssetsPath + "/Saves";

            while (true)
            {
                string xmainDir = mainDir + "/Save_" + UnityEngine.Random.Range(100, 1000).ToString();
                if (!Directory.Exists(xmainDir))
                {
                    Directory.CreateDirectory(xmainDir);
                    mainDir = xmainDir;
                    break;
                }
            }

            int howMany = (int)((float)team.Length - Mathf.Pow(team.Length, 1f / 3f));

            Mathf.Pow(team.Length, 0.33f);
            for (int i = howMany; i <= team.Length - 1; i++)
            {
                string name = "/Ag[" + i + "]_Fit[" + team[i].script.GetFitness().ToString("0.00") + "].txt";
                NeuralNetwork net = new NeuralNetwork(team[i].script.network);//Here was made a copy due to some weird write access error
                NeuralNetwork.WriteBrain(net, (mainDir + name));
            }
            string color = ColorConvertor.GetRichTextColorFromColor32(new Color32((byte)0, (byte)255, (byte)38, (byte)1));

            StringBuilder message = new StringBuilder();
            message.Append("<color=");
            message.Append(color);
            message.Append(">");
            message.Append((team.Length - howMany));
            message.Append(" neural networks have been saved in </color><i>");
            message.Append(mainDir);
            message.Append("</i>");
            Debug.Log(message.ToString());
        }
    }
    internal readonly struct Functions
    {
        //Activation
        static public float ActivationFunctionBinaryStep(float value)
        {
            if (value < 0)
                return 0;
            else return 1;
        }
        static public float ActivationFunctionSigmoid(float value)
        {
            //values range [0,1]
            // Function is x = 1/(1 + e^(-x))
            return (float)1f / (1f + Mathf.Exp(-value));
        }
        static public float ActivationFunctionTanh(float value)
        {
            return (float)System.Math.Tanh((double)value);
            /*
             //Other variant is to shift the sigmoid function
               return (float)2f / (1f + Mathf.Exp(-2*value)) - 1;
             
             
             */
        }
        static public float ActivationFunctionReLU(float value)
        {
            return Mathf.Max(0, value);
        }
        static public float ActivationFunctionLeakyReLU(float value, float alpha = 0.2f)
        {
            if (value > 0)
                return value;
            else return value * alpha;
        }
        static public float ActivationFunctionSiLU(float value)
        {
            return value * ActivationFunctionSigmoid(value);
        }
        static public void ActivationFunctionSoftMax(ref float[] values)
        {
            float sum = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = Mathf.Exp(values[i]);
                sum += values[i];
            }
            for (int i = 0; i < values.Length; i++)
            {
                values[i] /= sum;
            }
        }

        //Derivatives
        static public float DerivativeTanh(float value)
        {
            return 1f - (float)Math.Pow(Math.Tanh(value), 2);
        }
        static public float DerivativeSigmoid(float value)
        {
            return ActivationFunctionSigmoid(value) * (1 - ActivationFunctionSigmoid(value));
        }
        static public float DerivativeBinaryStep(float value)
        {
            return 0;
        }
        static public float DerivativeReLU(float value)
        {
            if (value < 0)
                return 0;
            else return 1;
        }
        static public float DerivativeLeakyReLU(float value, float alpha = 0.2f)
        {
            if (value < 0)
                return alpha;
            else return 1;
        }
        static public float DerivativeSiLU(float value)
        {
            return (1 + Mathf.Exp(-value) + value * Mathf.Exp(-value)) / Mathf.Pow((1 + Mathf.Exp(-value)), 2);
            //return ActivationFunctionSigmoid(value) * (1 + value * (1 - ActivationFunctionSigmoid(value))); -> works the same
        }
        static public void DerivativeSoftMax(ref float[] values)
        {
            float sum = 0f;

            foreach (float item in values)
                sum += Mathf.Exp(item);


            for (int i = 0; i < values.Length; i++)
            {
                float ePowI = Mathf.Exp(values[i]);
                values[i] = (ePowI * sum - ePowI * ePowI) / (sum * sum);
            }
        }


        //Cost Functions
        static public float QuadraticNodeCost(float outputActivation, float expectedOutput)
        {
            float error = outputActivation - expectedOutput;
            return error * error;
        }
        static public float QuadraticNodeCostDerivative(float outputActivation, float expectedOutput)
        {
            return 2 * (outputActivation - expectedOutput);
        }
        static public float AbsoluteNodeCost(float outputActivation, float expectedOutput)
        {
            return Mathf.Abs(outputActivation - expectedOutput);
        }
        static public float AbsoluteNodeCostDerivative(float outputActivation, float expectedOutput)
        {
            if ((outputActivation - expectedOutput) > 0)
                return 1;
            else return -1;
        }
        static public float CrossEntropyNodeCost(float outputActivation, float expectedOutput)
        {
            double v = (expectedOutput == 1) ? -System.Math.Log(outputActivation) : -System.Math.Log(1 - outputActivation);
            return (float)v;
        }
        static public float CrossEntropyNodeCostDerivative(float outputActivation, float expectedOutput)
        {
            if (outputActivation == 0 || expectedOutput == 1)
                return 0;
            return (-outputActivation + expectedOutput) / (outputActivation * (outputActivation - 1));
        }

        // Mutations
        static public void ClassicMutation(ref float weight)
        {
            float randNum = UnityEngine.Random.Range(0f, 10f);

            if (randNum <= 2f)//20% chance of flip sign of the weightOrBias
            {
                weight *= -1f;
            }
            else if (randNum <= 4f)//20% chance of fully randomize weightOrBias
            {
                weight = UnityEngine.Random.Range(-.5f, .5f);
            }
            else if (randNum <= 6f)//20% chance of increase to 100 - 200 %
            {
                float factor = UnityEngine.Random.value + 1f;
                weight *= factor;
            }
            else if (randNum <= 8f)//20% chance of decrease in range 0 - 100 %
            {
                float factor = UnityEngine.Random.value;
                weight *= factor;
            }
            else
            {
            }//20% chance of NO MUTATION

        }
        static public void LightPercentageMutation(ref float weight)
        {
            //increase/decrease all to a max of 50%
            float sign = UnityEngine.Random.value;
            float factor;
            if (sign > .5f)
            {
                factor = UnityEngine.Random.Range(1f, 1.5f);
            }
            else
            {
                factor = UnityEngine.Random.Range(.5f, 1f);
            }
            weight *= factor;
        }
        static public void StrongPercentagegMutation(ref float weight)
        {
            //increase/decrease all to a max of 100%

            float sign = UnityEngine.Random.value;
            float factor;
            if (sign > .5f)//increase
            {
                factor = UnityEngine.Random.value + 1f;
            }
            else//decrease
            {
                factor = UnityEngine.Random.value;

            }
            weight *= factor;

        }
        static public void LightValueMutation(ref float weight)
        {
            // + 0 -> .5f or  - 0 -> .5f
            float randNum = UnityEngine.Random.Range(-.5f, .5f);
            weight += randNum;
        }
        static public void StrongValueMutation(ref float weight)
        {
            float randNum = UnityEngine.Random.Range(-1f, 1f);
            weight += randNum;
        }
        static public void ChaoticMutation(ref float weight)
        {
            float chance = UnityEngine.Random.value;
            if (chance < .125f)
                weight = Functions.InitializationFunctionStandardNormal(0.15915f, 2f, 0.3373f);
            else if (chance < .3f)
                ClassicMutation(ref weight);
            else if (chance < .475f)
                LightPercentageMutation(ref weight);
            else if (chance < .65f)
                StrongPercentagegMutation(ref weight);
            else if (chance < .825f)
                LightValueMutation(ref weight);
            else
                StrongValueMutation(ref weight);

        }

        //Initialization 
        static public float InitializationFunctionStandardNormal(float l, float k, float z)
        {
            float x = UnityEngine.Random.value;
            float sign = UnityEngine.Random.value;
            if (sign > .5f)
                return (float)Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);
            else
                return (float)-Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);


        }
        static float RandomInNormalDistribution(System.Random rng, float mean, float standardDeviation)
        {
            float x1 = (float)(1 - rng.NextDouble());
            float x2 = (float)(1 - rng.NextDouble());

            float y1 = Mathf.Sqrt(-2.0f * Mathf.Log(x1)) * Mathf.Cos(2.0f * (float)Math.PI * x2);
            return y1 * standardDeviation + mean;
        }

        //Complementary
        static public void ConvertStrArrToIntArr(string[] str, ref int[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                try
                {
                    arr[i] = int.Parse(str[i]);
                }
                catch (System.Exception e)
                {
                    Debug.LogError(str[i] + " : " + e);
                }

            }
        }
        static public void ConvertStrArrToFloatArr(string[] str, ref float[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
            {
                try
                {
                    arr[i] = float.Parse(str[i]);
                }
                catch (System.Exception e)
                {
                    Debug.LogError(str[i] + " : " + e);
                }

            }
        }
        static public void Swap<T>(ref T[] objArray, int index1, int index2)
        {
            if (objArray == null && objArray.Length <= index1 && objArray.Length <= index2) return;

            var temp = objArray[index1];
            objArray[index1] = objArray[index2];
            objArray[index2] = temp;
        }
    }
    internal readonly struct ColorConvertor
    {
        public static void ConvertColorToColor32(Color color, ref Color32 color32)
        {
            color32.r = System.Convert.ToByte(color.r * 255f);
            color32.g = System.Convert.ToByte(color.g * 255f);
            color32.b = System.Convert.ToByte(color.b * 255f);
            color32.a = System.Convert.ToByte(color.a * 255f);
        }
        public static string GetRichTextColorFromColor32(Color32 color)
        {
            string clr = "#";
            clr += GetHexFrom(color.r);
            clr += GetHexFrom(color.g);
            clr += GetHexFrom(color.b);
            return clr;
        }
        public static string GetHexFrom(int value)
        {
            ///The format of the Number is returned in XX Format
            int firstValue = value;

            StringBuilder hexCode = new StringBuilder();
            int remainder;

            while (value > 0)
            {
                remainder = value % 16;
                value -= remainder;
                value /= 16;

                hexCode.Append(GetHexDigFromIntDig(remainder));
            }
            if (firstValue <= 15)
                hexCode.Append("0");
            if (firstValue == 0)//Case 0, we need to return 00
                hexCode.Append("0");

            string hex = hexCode.ToString();
            ReverseString(ref hex);
            return hex;
        }
        public static string GetHexDigFromIntDig(int value)
        {
            if (value < 0 || value > 15)
            {
                Debug.LogError("Value Parsed is not a Digit in HexaDecimal");
                return null;
            }
            if (value < 10)
                return value.ToString();
            else if (value == 10)
                return "A";
            else if (value == 11)
                return "B";
            else if (value == 12)
                return "C";
            else if (value == 13)
                return "D";
            else if (value == 14)
                return "E";
            else if (value == 15)
                return "F";
            else return null;
        }
        public static void ReverseString(ref string str)
        {
            char[] charArray = str.ToCharArray();
            System.Array.Reverse(charArray);
            str = new string(charArray);
        }
    }
    class JSONSaver
    {   //Not used until multi-dimensional arrays can be serialized as json
        string path;
        object obj;

        public JSONSaver(object @object, string path)
        {
            this.path = path;
            path += ".json";
            this.obj = @object;
        }
        public void Save()
        {
            string json = JsonUtility.ToJson(obj);
            System.IO.File.WriteAllText(path, json);
        }
        public void Load()
        {
            string json = System.IO.File.ReadAllText(path);
            object obj = JsonUtility.FromJson<object>(json);
        }
        public object GetObject()
        {
            return obj;
        }
        public void ChangePath(string newPath)
        {
            this.path = newPath;
        }


    }

    public struct AI
    {
        /// <summary>
        /// Agent gameobject.
        /// </summary>
        public GameObject agent;
        /// <summary>
        /// Agent script component of your agent.
        /// </summary>
        public Agent script;
        /// <summary>
        /// Current agent fitness of your agent.
        /// </summary>
        public float fitness;
    }
    public struct PosAndRot
    {
        public Vector3 position, scale;
        public Quaternion rotation;
        public PosAndRot(Vector3 pos, Vector3 scl, Quaternion rot)
        {
            position = pos;
            scale = scl;
            rotation = rot;
        }
        public PosAndRot(UnityEngine.Transform transform)
        {
            position = transform.position;
            scale = transform.localScale;
            rotation = transform.rotation;
        }
    }
    public struct SensorBuffer
    {
        private float[] buffer;
        private int sizeIndex;
        public SensorBuffer(int capacity)
        {
            buffer = new float[capacity];
            for (int i = 0; i < capacity; i++)
                buffer[i] = 0;
            sizeIndex = 0;
        }
        public float[] GetBuffer()
        {
            return buffer;
        }
        public int GetBufferCapacity()
        {
            if (buffer == null)
                return 0;
            else return buffer.Length;
        }


        /// <summary>
        /// Appends a float observation to the SensorBuffer.
        /// </summary>
        /// <param name="observation1">Value of the observation</param>
        public void AddObservation(float observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        public void AddObservation(int observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        public void AddObservation(uint observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full. Increase the space size or remove this observation.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        public void AddObservation(Vector2 observation2)
        {
            if (buffer.Length - sizeIndex < 2)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector2 observation of size 2 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation2.x;
            buffer[sizeIndex++] = observation2.y;
        }
        public void AddObservation(Vector3 observation3)
        {

            if (buffer.Length - sizeIndex < 3)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector3 observation of size 3 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation3.x;
            buffer[sizeIndex++] = observation3.y;
            buffer[sizeIndex++] = observation3.z;
        }
        public void AddObservation(Vector4 observation4)
        {

            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Vector4 observation of size 4 is too large.");
                return;
            }

            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        public void AddObservation(Quaternion observation4)
        {
            if (buffer.Length - sizeIndex < 4)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Quaternion observation of size 4 is too large.");
                return;
            }
            buffer[sizeIndex++] = observation4.x;
            buffer[sizeIndex++] = observation4.y;
            buffer[sizeIndex++] = observation4.z;
            buffer[sizeIndex++] = observation4.w;
        }
        public void AddObservation(UnityEngine.Transform obsevation10)
        {
            if (buffer.Length - sizeIndex < 10)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Transform observation of size 10 is too large.");
                return;
            }
            AddObservation(obsevation10.position);
            AddObservation(obsevation10.localScale);
            AddObservation(obsevation10.rotation);
        }
        public void AddObservation(float[] observations1)
        {
            if (buffer.Length - sizeIndex < observations1.Length)
            {
                Debug.Log("SensorBuffer available space is " + (buffer.Length - sizeIndex) + ". Float array observations is too large.");
                return;
            }
            foreach (var item in observations1)
            {
                AddObservation(item);
            }
        }
    }
    public struct ActionBuffer
    {
        private float[] buffer;
        public ActionBuffer(float[] actions)
        {
            buffer = actions;
        }
        public ActionBuffer(int capacity)
        {
            buffer = new float[capacity];
        }

        /// <summary>
        /// Get the buffer array with every action values.
        /// <para>Can be used instead of using GetAction() method.</para>
        /// </summary>
        /// <returns>float[] copy of the buffer</returns>
        public float[] GetBuffer()
        {
            return buffer;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <returns>Total actions number</returns>
        public int GetBufferCapacity()
        {
            return buffer != null ? buffer.Length : 0;
        }
        /// <summary>
        /// Returns the value from the index parameter.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer.</param>
        /// <returns>float</returns>
        public float GetAction(uint index)
        {
            try
            {
                return buffer[index];
            }
            catch { Debug.LogError("Action index out of range."); }
            return 0;
        }
        /// <summary>
        /// Sets the action from ActionBuffer with a specific value.
        /// </summary>
        /// <param name="index">The index of the action from ActionBuffer</param>
        /// <param name="action1">The value of the action to be set</param>
        public void SetAction(uint index, float action1)
        {
            buffer[index] = action1;
        }
        /// <summary>
        /// Returns the index of the max value from ActionBuffer.
        /// <para>Usually used when SoftMax is the output activation function.</para>
        /// </summary>
        /// <returns>The index or -1 if all elements are equal.</returns>
        public int GetIndexOfMaxValue()
        {
            float max = float.MinValue;
            int index = -1;
            bool equal = true;
            for (int i = 0; i < buffer.Length; i++)
            {
                if (i > 0 && buffer[i] != buffer[i - 1])
                    equal = false;

                if (buffer[i] > max)
                {
                    max = buffer[i];
                    index = i;
                }
            }
            return equal == true ? -1 : index;

        }
    }
    internal struct Node
    {
        public float valueIn,//before activation
                     valueOut,//after activation
                     costValue;
    }
    internal struct Sample
    {
        public float[] inputs;
        public float[] expectedOutputs;
    }

    public enum TrainingType
    {
        [Tooltip("@static environment\n@single environment\n@multiple agents")]
        NotSpecified,

        //Agents overlap eachother, environmental objects are common
        [Tooltip("@agents are overlapping in the same environment(s)\nif no start, agent model is used as a starting position")]
        MoreAgentsPerEnvironment,

        //Agents train separately, environmental objects are personal for each agent
        [Tooltip("@one agent per each environment found\nusually used for letting just 1 agent interact with the environment")]
        OneAgentPerEnvironment,
    }
    public enum BehaviorType
    {
        [Tooltip("Doesn't move")]
        Static,
        [Tooltip("Can move only by user input\n@override Heuristic()\n@override OnActionReceived()")]
        Manual,
        [Tooltip("Moves independently\n@override CollectObservations()\n@override OnActionReceived()")]
        Self,
        [Tooltip("Trains by user input\nNo Trainer required\n@override CollectObservations()\n@override Heuristic()\n@override OnActionReceived()")]
        Heuristic,


    }
    public enum TrainingStrategy
    {
        [Tooltip("@(1/2) best AI reproduce\n@(1/2) copies + mutated")]
        Strategy1,
        [Tooltip("@(1/3) best AI reproduce\n@(1/3)copies + mutation\n@(1/3) worst AI get best brain + mutation")]
        Strategy2,
        [Tooltip("@(1)best AI reproduce\n@(Rest) copies + mutation")]
        Strategy3,

    }
    public enum MutationStrategy
    {
        [Tooltip("20% -> * (-1) " +
            "\n20% -> +.5f | -.5f" +
            "\n20% -> + 0%~100%" +
            "\n20% -> - 0%~100%" +
            "\n20% -> no mutation")]
        Classic,
        [Tooltip("50% -> -(0%~50%)" +
            "\n50% -> +(0%~50%)" +
            "\n@no sign change" +
            "\n@best for finetuning")]
        LightPercentage,
        [Tooltip("50% -> -(0f~.5f)" +
            "\n50% -> +(0f~.5f)")]
        LightValue,
        [Tooltip("50% -> -(0%~100%)" +
            "\n50% -> +(0%~100%)" +
            "\n@no sign change" +
            "\n@best for deeptuning")]
        StrongPercentage,
        [Tooltip("50% -> -(0f~1f)" +
                  "\n50% -> +(0f~1f)")]
        StrongValue,
        [Tooltip("12.5% -> New value from normal distribution" +
            "\n17.5% -> Classic mutation" +
            "\n17.5% -> LightPercentage mutation" +
            "\n17.5% -> LightValue mutation" +
            "\n17.5% -> StrongPercentage mutation" +
            "\n17.5% -> StrongValue mutation")]
        Chaotic,

    }
    public enum ActivationFunctionType
    {
        //NO REAL TIME MODIFICATION
        [Tooltip("@output: 0 or 1\n" +
                 "@good for output layer - binary value")]
        BinaryStep,
        [Tooltip("@output: (0, 1)\n" +
                 "@good for output layer - good value range (positive)")]
        Sigmoid,
        [Tooltip("@output: (-1, 1)\n" +
                 "@best for output layer - good value range")]
        Tanh,
        [Tooltip("@output: [0, +inf)\n" +
                 "@good for hidden layers - low computation")]
        Relu,
        [Tooltip("@output: (-inf*,+inf)\n" +
                 "@best for hidden layers - low computation")]
        LeakyRelu,
        [Tooltip("@output: [-0.278, +inf)\n" +
                 "@smooth ReLU - higher computation")]
        Silu,
        [Tooltip("@output: [0, 1]\n" +
                 "@output activation ONLY\n" +
                 "@good for decisional output")]
        SoftMax,

    }
    public enum InitializationFunctionType
    {
        [Tooltip("@value: [0, 1]")]
        RandomValue,
        [Tooltip("@value: average 0.725\n" +
            "@l = 0.15915f\n" +
            "@k = 2f\n" +
            "@z = 0.3373f")]
        StandardNormal1,
        [Tooltip("@value: average 0.673\n" +
            "@l = 0.15915f\n" +
            "@k = 1.061f\n" +
            "@z = 0.3373f")]
        StandardNormal2,
        [Tooltip("@value: average 1.065\n" +
            "@l = 0.15915f\n" +
            "@k = 2f\n" +
            "@z = 0.722")]
        StandardNormal3
    }
    public enum LossFunctionType
    {
        [Tooltip("(output - expectedOutput)^2")]
        Quadratic,
        [Tooltip("abs(output - expectedOutput)")]
        Absolute,
        CrossEntropy,
    }
    public enum HeuristicModule
    {
        [Tooltip("@append data to the file below\n@creates a file if doesn't exist")]
        Append,
        [Tooltip("@overwrite data to the file below\n@creates a file if doesn't exist")]
        Write,
        [Tooltip("@use data from the file below")]
        Learn,
    }
}
