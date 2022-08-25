using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.IO;
using UnityEngine.UI;
using System.IO;
using System.Text;
using System.Linq;
using TMPro;
using UnityEditor;
using System;
using System.Linq.Expressions;
using Unity.VisualScripting;
using Unity.Burst;
using UnityEngine.UIElements;
namespace MLFramework
{
    public class NeuralNetwork
    {
        public static ActivationFunctionType activation = ActivationFunctionType.Tanh;
        public static ActivationFunctionType outputActivation = ActivationFunctionType.Tanh;
        public static MutationStrategy mutation = MutationStrategy.Classic;
        public static InitializationFunctionType initialization = InitializationFunctionType.StandardNormal1;

        protected int[] layers;
        protected float[][] neurons;
        protected float[][][] weights;
        protected float[][] biases;//They are added for each neuron like as a separate weightOrBias without a neuron. All but input neurons have bias

        protected float fitness;

        public NeuralNetwork(int[] layers)
        {
            InitializeLayers(layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            fitness = 0f;

        }//basic constructor
        public NeuralNetwork(NeuralNetwork copyNN)
        {
            InitializeLayers(copyNN.layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            SetWeightsWith(copyNN.weights);
            SetBiasesWith(copyNN.biases);
        }//construct from copy
        public NeuralNetwork(string path)
        {
            if (new FileInfo(path).Length == 0)
            {
                Debug.LogError("The training cannot start! Reason: Brain Model uploaded file is empty");
                return;
            }
            //For each line, there is 1 more element that was read when splitting, so kill it everywhere
            List<string> fileLines = File.ReadAllLines(path).ToList();

            //Get Layers Data
            string[] layersLineStr = fileLines[0].Split("n,");//one more read than neccesary
            int noLayers = layersLineStr.Length - 1;
            int[] layersLineInt = new int[noLayers];
            ConvertStrArrToIntArr(layersLineStr, ref layersLineInt);
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
                ConvertStrArrToFloatArr(lineStr, ref lineInt);


                //This array must be devided depeding on the previous neuronsNumber number of neurons
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
                ConvertStrArrToFloatArr(lineStr, ref lineInt);
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
                for (int i = 0; i < biasesList.Count; i++)
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
            if (initialization == InitializationFunctionType.StandardNormal1)
            {
                for (int i = 0; i < axons.Length; i++)
                {
                    axons[i] = InitializationFunctionStandardNormal();
                }
            }
            else if (initialization == InitializationFunctionType.RandomValue)
            {
                for (int i = 0; i < axons.Length; i++)
                {
                    axons[i] = InitializationFunctionRandomvalue();
                }
            }

        }

        //-------------------PROPAGATION--------------------//
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
                    if (l == layers.Length - 1)
                        neurons[l][n] = Activate(value, true);
                    else
                        neurons[l][n] = Activate(value, false);
                }
            }



            return neurons[neurons.Length - 1]; //Return the last neuronsNumber (OUTPUT)
        }
        public float[] CalculateOutputsAndWeightedInputs(float[] inputs, ref float[][] weightedInputs)
        {
            SetInputs(inputs);

            for (int l = 1; l < layers.Length; l++)
            {
                weightedInputs[l] = new float[neurons[l].Length];
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    float value = biases[l][n];
                    for (int p = 0; p < neurons[l - 1].Length; p++)
                    {
                        value += weights[l - 1][n][p] * neurons[l - 1][p];
                    }
                    weightedInputs[l][n] = value;

                    if (l == layers.Length - 1)
                        neurons[l][n] = Activate(value, true);
                    else
                        neurons[l][n] = Activate(value, false);



                }
            }



            return neurons[neurons.Length - 1]; //Return the last neuronsNumber (OUTPUT)
        }

        //--------------------MUTATIONS---------------------//
        public void MutateWeightsAndBiases()
        {
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    for (int k = 0; k < weights[i][j].Length; k++)
                        MutateWeightOrBias(ref weights[i][j][k]);

            for (int i = 0; i < biases.Length; i++)
                for (int j = 0; j < biases[i].Length; j++)
                    MutateWeightOrBias(ref biases[i][j]);
        }
        protected void MutateWeightOrBias(ref float weightOrBias)
        {
            if (mutation == MutationStrategy.Classic)
                ClassicMutation(ref weightOrBias);

            else if (mutation == MutationStrategy.LightPercentage)
                LightPercentageMutation(ref weightOrBias);
            else if (mutation == MutationStrategy.LightValue)
                LightValueMutation(ref weightOrBias);

            else if (mutation == MutationStrategy.StrongPercentage)
                StrongPercentagegMutation(ref weightOrBias);
            else if (mutation == MutationStrategy.StrongValue)
                StrongValueMutation(ref weightOrBias);

            else if (mutation == MutationStrategy.Chaotic)
                ChaoticMutation(ref weightOrBias);
        }

        //------------------MUTATION STRATEGIES------------//
        void ClassicMutation(ref float weight)
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
        void LightPercentageMutation(ref float weight)
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
        void StrongPercentagegMutation(ref float weight)
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
        void LightValueMutation(ref float weight)
        {
            // + 0 -> .5f or  - 0 -> .5f
            float randNum = UnityEngine.Random.Range(-.5f, .5f);
            weight += randNum;
        }
        void StrongValueMutation(ref float weight)
        {
            float randNum = UnityEngine.Random.Range(-1f, 1f);
            weight += randNum;
        }
        void ChaoticMutation(ref float weight)
        {
            float chance = UnityEngine.Random.value;
            if (chance < .125f)
                weight = InitializationFunctionStandardNormal();
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
        public int GetInputsNumber()
        {
            return layers[0];
        }
        public int GetOutputsNumber()
        {
            return layers[layers.Length - 1];
        }

        //------------------------ACTIVATION FUNCTIONS-----=-------------------//
        static float Activate(float value, bool outputLayer = false)
        {
            if (outputLayer == false)
            {
                if (activation == ActivationFunctionType.Sigmoid)
                    return ActivationFunctionSigmoid(value);
                else if (activation == ActivationFunctionType.Tanh)
                    return ActivationFunctionHyperbolicTangent(value);
                else if (activation == ActivationFunctionType.ReLU)
                    return ActivationFunctionRectifiedLinearUnit(value);
                else if (activation == ActivationFunctionType.LeakyReLU)
                    return ActivationFunctionLeakyRectifiedLinearUnit(value);
                else if (activation == ActivationFunctionType.SiLU)
                    return ActivationFunctionSoftPlus(value);
                else if (activation == ActivationFunctionType.BinaryStep)
                    return ActivationFunctionBinaryStep(value);
            }
            else
            {
                if (outputActivation == ActivationFunctionType.Sigmoid)
                    return ActivationFunctionSigmoid(value);
                else if (outputActivation == ActivationFunctionType.Tanh)
                    return ActivationFunctionHyperbolicTangent(value);
                else if (outputActivation == ActivationFunctionType.ReLU)
                    return ActivationFunctionRectifiedLinearUnit(value);
                else if (activation == ActivationFunctionType.LeakyReLU)
                    return ActivationFunctionLeakyRectifiedLinearUnit(value);
                else if (outputActivation == ActivationFunctionType.SiLU)
                    return ActivationFunctionSoftPlus(value);
                else if (activation == ActivationFunctionType.BinaryStep)
                    return ActivationFunctionBinaryStep(value);

            }
            return 0f;
        }
        static float ActivationFunctionBinaryStep(float value)
        {
            if (value < 0)
                return 0;
            else return 1;
        }
        static float ActivationFunctionSigmoid(float value)
        {
            //values range [0,1]
            // Function is x = 1/(1 + e^(-x))
            return (float)1f / (1f + Mathf.Exp(-value));
        }
        static float ActivationFunctionHyperbolicTangent(float value)
        {
            return (float)System.Math.Tanh((double)value);
            /*
             //Other variant is to shift the sigmoid function
               return (float)2f / (1f + Mathf.Exp(-2*value)) - 1;
             
             
             */
        }
        static float ActivationFunctionRectifiedLinearUnit(float value)
        {
            return Mathf.Max(0, value);
        }
        static float ActivationFunctionLeakyRectifiedLinearUnit(float value, float alpha = 0.2f)
        {
            if (value > 0)
                return value;
            else return value * alpha;
        }
        static float ActivationFunctionSoftPlus(float value)
        {
            return Mathf.Log(1 + Mathf.Exp(value));
        }

        //-----------------------INITIALIZATION FUNCTIONS----------------------//
        static float InitializationFunctionStandardNormal(float l = 0.15915f, float k = 2f, float z = 0.3373f)
        {
            float x = UnityEngine.Random.value;
            float sign = UnityEngine.Random.value;
            if (sign > .5f)
                return (float)Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);
            else
                return (float)-Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);


        }
        static float InitializationFunctionRandomvalue()
        {
            return (float)UnityEngine.Random.value;
        }
        /* static float NormalDistributionOf(float x, float sigma = 1f, float mu = 0f)
      {
          return (float)(1 / System.Math.Sqrt((2f * System.Math.PI * Mathf.Pow(sigma, 2f)))
                             * Mathf.Exp(-1f / 2f * Mathf.Pow((x - mu) / sigma, 2f)));
      }*/

        //--------------COMPLEMENTARY METHODS---------------//
        protected void SetInputs(float[] inputs)
        {
            if (inputs.Length != neurons[0].Length)
                Debug.Log("The number of inputs received is " + inputs.Length + " and the number of input neurons is " + neurons[0].Length + ". Expect some inputs to be ignored!");

            for (int i = 0; i < neurons[0].Length; i++)
            {
                neurons[0][i] = inputs[i];
            }
        }
        private void ConvertStrArrToIntArr(string[] str, ref int[] arr)
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


        //-------------------HEURISTIC ----------------------//
        #region HEURISTIC
        public void ApplyGradientsToLayer(int layerIndex, float[] biasLayerGradient, float[][] weightLayerGradient, float learnRate)
        {
            for (int neuronIndex = 0; neuronIndex < layers[layerIndex]; neuronIndex++)
            {
                biases[layerIndex - 1][neuronIndex] -= biasLayerGradient[neuronIndex] * learnRate;
                for (int weightIndex = 0; weightIndex < layers[layerIndex - 1]; weightIndex++)
                    weights[layerIndex - 1][neuronIndex][weightIndex] -= weightLayerGradient[neuronIndex][weightIndex] * learnRate;
            }
        }
        #endregion


    }
    public class AgentBase : UnityEngine.MonoBehaviour
    {
        [Header("===== Agent Properties =====")]
        public BehaviorType behavior = BehaviorType.Static;
        [Tooltip("Assign a path to a brain model")] public string path = null;
        [Tooltip("@if has brain: saves current brain data\n@else: creates a brain using Network Properties.\n@folder: StreamingAssets/Neural_Networks.")] public bool SaveBrain = false;
        public NeuralNetwork network = null;
        List<PosAndRot> initialPosition = new List<PosAndRot>(); static int parseCounter = 0;

        [Header("===== Network Properties =====")]
        [SerializeField, Range(1, 50), Tooltip("The number of Inputs that the Agent will receive [-1,1]")] private int sensorSize = 2;
        [SerializeField, Range(1, 50), Tooltip("The number of Outputs that the Agent will return [-1,1]")] private int actionSize = 2;
        [SerializeField, Tooltip("Each element is a hidden layer\nEach value is the number of neurons\n@biases not count")] private List<uint> hiddenLayers;

        [Header("===== Heuristic Properties =====")]
        [Range(0, 1), Tooltip("TIP: keep the value resonable low")] public float learnRate = 0.01f;
        [SerializeField, Tooltip("@determines loss error form")] private LossFunctionType costType = LossFunctionType.Quadratic;
        [SerializeField, Tooltip("@read only")] private float currentError;


        protected virtual void Awake()
        {
            GetAllTransforms();
            //On heuristic and self the brains are made directly in HeuristicAction and SelfAction,this may cause in action lag

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
                    Debug.LogError("Cannot Self Control because the Brain Path uploaded is invalid");
                    return;
                }
                this.network = new NeuralNetwork(path);
            }
            SensorBuffer sensorBuffer = new SensorBuffer(network.GetLayers()[0]);
            CollectObservations(ref sensorBuffer);
            ActionBuffer actionBuffer = new ActionBuffer(network.ForwardPropagation(sensorBuffer.GetBuffer()));
            OnActionReceived(in actionBuffer);
        }
        void ManualAction()
        {
            ActionBuffer buffer = new ActionBuffer(50);
            Heuristic(ref buffer);
            OnActionReceived(in buffer);
        }
        void HeuristicAction()
        {
            if (network == null)
            {
                if (path == null || path == "" || new FileInfo(path).Length == 0)
                {
                    Debug.LogError("Cannot train Heuristicaly because the Brain Path uploaded is invalid");
                    return;
                }
                this.network = new NeuralNetwork(path);
            }
            //Get inputs
            SensorBuffer inputs = new SensorBuffer(network.GetLayers()[0]);
            CollectObservations(ref inputs);
            //Get outputs
            ActionBuffer propagatedOutputs = new ActionBuffer(network.ForwardPropagation(inputs.GetBuffer()));
            //Get userOutputs
            ActionBuffer desiredOutputs = new ActionBuffer(propagatedOutputs.GetBufferCapacity());
            Heuristic(ref desiredOutputs);
            OnActionReceived(desiredOutputs);



            //So we have:
            // - inputs
            // - outputs from propagation
            // - outputs desired
            // - learnRate

            //GO FOR:
            //network.HeuristicTrain(in inputs, in propagatedOutputs, desiredOutputs, learnRate);

            ///IMPLEMENT HEURISTIC TRAINING IN NEURAL NETWORK
            ///create batches or train real time by the data
        }
        //-------------------------------------------FOR USE WHEN OVERRIDING----------------------------------------------//
        protected virtual void Heuristic(ref ActionBuffer actionsOut)
        {

        }
        protected virtual void CollectObservations(ref SensorBuffer sensorBuffer)
        {

        }
        protected virtual void OnActionReceived(in ActionBuffer actionBuffer)
        {

        }

        //-------------------------------------------FOR USE WHEN TRAINING-------------------------------------------------//
        public void AddReward(float reward, bool evenIfActionEnded = false)
        {
            if (behavior == BehaviorType.Manual)
                return;
            if (evenIfActionEnded == false && behavior == BehaviorType.Static)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot add reward because neural network is null");
                return;
            }
            network.AddFitness(reward);
        }
        public void SetReward(float reward, bool evenIfActionEnded = false)
        {
            if (behavior == BehaviorType.Manual)
                return;
            if (evenIfActionEnded == false && behavior == BehaviorType.Static)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot set reward because neural network is null");
                return;
            }
            network.SetFitness(reward);
        }
        public void EndAction()
        {
            if (behavior == BehaviorType.Self)
                behavior = BehaviorType.Static;
            else if (behavior == BehaviorType.Manual || behavior == BehaviorType.Heuristic)
                ResetToInitialPosition();
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
        static private void ApplyTransformTo(ref GameObject obj, PosAndRot trnsfrm)
        {
            obj.transform.position = trnsfrm.position;
            obj.transform.localScale = trnsfrm.scale;
            obj.transform.rotation = trnsfrm.rotation;
        }

        //--------------------------------------SETTERS AND GETTERS--------------------------------------------//
        public void SetFitnessTo(float value)
        {
            this.network.SetFitness(0);
        }
        public float GetFitness()
        {
            if (network != null)
                return network.GetFitness();
            else return 0f;
        }
        public string GetPath()
        {
            StringBuilder pathsb = new StringBuilder();
            pathsb.Append(Application.streamingAssetsPath);
            pathsb.Append("/Neural_Networks/");

            if (!Directory.Exists(pathsb.ToString()))
                Directory.CreateDirectory(pathsb.ToString());

            pathsb.Append("NeuralNetworkID");
            pathsb.Append(((int)this.gameObject.GetInstanceID()) * (-1));
            pathsb.Append(".txt");


            return pathsb.ToString();
        }
        public List<PosAndRot> GetInitialPosition()
        {
            return initialPosition;
        }

        //---------------------------------------------BUTTONS---------------------------------------------------//
        void BUTTONSaveBrain()
        {
            if (SaveBrain == true)
            {
                SaveBrain = false;
                if (network != null)
                    NeuralNetwork.WriteBrain(network, GetPath());
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
                    NeuralNetwork.WriteBrain(network, GetPath());
                }

            }
        }

        //----------------------------------------HEURISTIC TRAINING--------------------------------------//
        #region Heuristic
        private float[][][] costGradientW;
        private float[][] costGradientB; bool initvars = false;
        private int[] netLayers;
        private void HeuristicTraining()
        {
            if (!initvars)
                InitVars();
            Learn();
        }
        private void InitVars()
        {
            if (network == null)
            {
                Debug.Log("Cannot start Heuristic Training. Network is missing!");
                behavior = BehaviorType.Static;
                return;
            }
            netLayers = network.GetLayers();
            int size = netLayers.Length - 1;
            costGradientW = new float[size][][];
            costGradientB = new float[size][];





            initvars = true;
        }
        private void Learn()
        {

            //BackPropagation();

            for (int i = 0; i < netLayers.Length - 1; i++)
            {
                network.ApplyGradientsToLayer(i, costGradientB[i], costGradientW[i], learnRate);
            }


            //Reset gradients array for the next batch
            for (int i = 0; i < costGradientW.Length; i++)
            {
                for (int j = 0; j < costGradientW[i].Length; j++)
                {
                    for (int k = 0; k < costGradientW[i][j].Length; k++)
                    {
                        costGradientW[i][j][k] = 0;
                    }
                }
            }
            for (int i = 0; i < costGradientB.Length; i++)
            {
                for (int j = 0; j < costGradientB[i].Length; j++)
                {
                    costGradientB[i][j] = 0;
                }
            }
        }

        float Cost()
        {
            int[] layers = network.GetLayers();
            // CollectInputs
            SensorBuffer inputs = new SensorBuffer(layers[0]);
            CollectObservations(ref inputs);
            // CollectOutputs
            float[] normalOutputs = network.ForwardPropagation(inputs.GetBuffer());//------------------------------outputs
            //CollectDesiredOuputs

            ActionBuffer desiredOutputsBuffer = new ActionBuffer(layers[layers.Length - 1]);
            Heuristic(ref desiredOutputsBuffer);

            float[] desiredOutputs = desiredOutputsBuffer.GetBuffer();//------------------------------------desired outputs


            float cost = 0;
            for (int nodeOut = 0; nodeOut < normalOutputs.Length; nodeOut++)
            {
                cost += NeuronCost(normalOutputs[nodeOut], desiredOutputs[nodeOut]);
            }
            return cost;
        }
        private void BackPropagation(float[] inputs, float[] expectedOutputs)
        {
            float[][] weightedInputs = new float[netLayers[netLayers.Length - 1]][];
            float[] outputs = network.CalculateOutputsAndWeightedInputs(inputs, ref weightedInputs);
            float[] neuronValues = CalculateOutputLayerNeuronsValues(outputs, expectedOutputs, weightedInputs[weightedInputs.Length - 1]);
            UpdateGradients(netLayers[netLayers.Length - 1], inputs, neuronValues);

            for (int i = 2; i < netLayers.Length - 1; i++)
            {
                neuronValues = CalculateHiddenLayerNeuronsValues(netLayers[netLayers.Length - 2], neuronValues);
                UpdateGradients(netLayers[netLayers.Length - i], inputs, neuronValues);
            }

        }
        private void UpdateGradients(int layerIndex, float[] inputs, float[] neuronValues)
        {
            for (int i = 0; i < netLayers[layerIndex]; i++)
            {
                for (int j = 0; j < netLayers[layerIndex - 1]; j++)
                {
                    float derivativeCosWrtWeight = inputs[j] * neuronValues[i];
                    costGradientW[layerIndex - 1][j][i] += derivativeCosWrtWeight;
                }
                float derivativeCostWrtBias = neuronValues[i];
                costGradientB[layerIndex - 1][i] += derivativeCostWrtBias;
            }
        }

        private float[] CalculateOutputLayerNeuronsValues(float[] activationOutputs, float[] expectedOutputs, float[] weightedInputs)
        {
            float[] neuronValues = new float[expectedOutputs.Length];
            for (int i = 0; i < neuronValues.Length; i++)
            {
                float costDerivative = NeuronCostDerivative(activationOutputs[i], expectedOutputs[i]);
                float activationDerivative = TanhDerivative(weightedInputs[i]);
                neuronValues[i] = costDerivative * activationDerivative;
            }


            return neuronValues;
        }
        private float[] CalculateHiddenLayerNeuronsValues(int oldLayerIndex, float[] oldNeuronValues)
        {
            float[] newNodeValues = new float[netLayers[oldLayerIndex]];
            for (int newNeuronIndex = 0; newNeuronIndex < netLayers[oldLayerIndex]; newNeuronIndex++)
            {
                float newNeuronValue = 2;
                for (int oldNeuronIndex = 0; oldNeuronIndex < netLayers[oldLayerIndex - 1]; oldNeuronIndex++)
                { //------------------------------Need implement
                    newNeuronValue = oldNeuronIndex;

                }
            }
            return newNodeValues;
        }

        float NeuronCost(float outputActivation, float expectedOutput)
        {
            float error = outputActivation - expectedOutput;
            return error * error;
        }
        float NeuronCostDerivative(float outputActivation, float expectedOutput)
        {
            return 2 * (outputActivation - expectedOutput);
        }
        private float Tanh(float x)
        {
            return 2f / (1 + Mathf.Exp((-2f) * x)) - 1f;
        }
        private float TanhDerivative(float x)
        {
            return 1f - (float)Math.Pow(Math.Tanh(x), 2);
        }
        #endregion
    }
    public class TrainerBase : UnityEngine.MonoBehaviour
    {
        [Header("===== Models =====")]
        [Tooltip("Agent model gameObject used as the ai")] public GameObject AIModel;
        [Tooltip("Brain model used to start the training with")] public string brainModelPath;
        [Tooltip("The model used updates in the first next generation\n@tip: use a copy of the brain")] public bool resetBrainModelFitness = false;
        [Tooltip("@resets the dynamic environmental object's positions")] public TrainingEnvironment environmentType = TrainingEnvironment.NotSpecified;
        [Tooltip("@save networks of best Ai's before moving to the next generation.\n@number of saves = sqrt(Team Size).\n@folder: /Saves/.")] public bool saveBrains = false;

        [Header("===== Statistics Display =====")]
        [Tooltip("First ObjectOfType<Camera>")] public bool cameraFollowsBestAI = true; GameObject cam; bool isOrtographic; Vector3 perspectiveOffset;
        [Tooltip("Load a Canvas TMPro to watch the current performance of AI's")] public TMPro.TMP_Text Labels = null;
        [Tooltip("Load a Canvas RectTransform to watch a Gizmos graph in SceneEditor")] public RectTransform Graph = null;
        List<float> bestResults;//memorize best results for every episode
        List<float> averageResults;//memorize avg results for every episode

        [Space, Header("===== Training Settings =====")]
        [Range(3, 300)] public int teamSize = 5;//IT cannot be 1 or 2, otherwise strategies will not work
        [Range(1, 10), Tooltip("Episodes needed to run until passing to the next Generation\n@TIP: divide the reward given by this number")] public int episodesPerEvolution = 1;
        [Range(1, 1000), Tooltip("Total Episodes in this Training Session")] public int maxEpisodes = 100; private int currentEpisode = 1;
        [Range(1, 1000), Tooltip("Maximum time allowed per Episode")] public float maxTimePerEpisode = 100f; float timeLeft;

        [Space, Header("===== Strategies =====")]
        [Tooltip("@in the beggining use Strategy1.\n@if AI's performance decreases, switch to Strategy2.\n@finetune the final Brain using Strategy3.")]
        public TrainingStrategy trainingStrategy = TrainingStrategy.Strategy1;
        [Tooltip("@mutates the weights and biases following certain rules")]
        public MutationStrategy mutationStrategy = MutationStrategy.Classic;
        [Tooltip("@always use Tanh in Heuristic training")]
        public ActivationFunctionType activationType = ActivationFunctionType.Tanh;
        [Tooltip("@influences the actionBuffer values")]
        public ActivationFunctionType outputActivationType = ActivationFunctionType.Tanh;
        [Tooltip("@initializes weights and biases of a newly created network")]
        public InitializationFunctionType initializationType = InitializationFunctionType.StandardNormal1;



        private NeuralNetwork modelNet;
        protected AI[] team;

        private int environmentsNumber = 0; private GameObject[] Environments;
        private int currentEnvironment = 0;
        protected List<PosAndRot>[] environmentsInitialTransform;//Every list is all transforms of a single environment
        protected List<PosAndRot>[] agentsInitialTransform;//Every list is all transforms of a single agent, Both use the same index
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
            NetworkChanges();
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
                Debug.LogError("The training cannot start! Reason: No AI Model uploaded");
                return false;
            }
            if (AIModel.GetComponent<Agent>() == null)
            {
                Debug.LogError("The training cannot start! Reason: AI Model is not a Agent");
                return false;
            }
            if (brainModelPath == null || brainModelPath == "")
            {
                Debug.LogError("The training cannot start! Reason: Brain Model not uploaded");
                return false;
            }
            if (!File.Exists(brainModelPath))
            {
                Debug.LogError("The training cannot start! Reason: Brain Model Path uploaded doesn't exists");
                return false;
            }
            modelNet = new NeuralNetwork(brainModelPath);
            if (resetBrainModelFitness)
                modelNet.SetFitness(0f);
            return true;
        }
        protected virtual void SetupTeam()
        {
            //Instatiate AI
            team = new AI[teamSize];
            for (int i = 0; i < team.Length; i++)
            {
                GameObject member = Instantiate(AIModel, AIModel.transform.position, Quaternion.identity);
                team[i].agent = member;
                team[i].agent.SetActive(true);
                team[i].script = member.GetComponent<Agent>() as Agent;
                team[i].fitness = 0f;
            }

            //Initialize AI
            for (int i = 0; i < team.Length; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(brainModelPath);
                script.SetFitnessTo(0f);

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
                Debug.Log("Training Session Ended!");
                foreach (var item in team)
                    item.script.behavior = BehaviorType.Static;
                startTraining = false;
            }
            UpdateDisplay();
        }
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
            {
                OnEpisodeEnd(ref team[i]);
            }
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

            NextEnvironment();
            ResetEnvironmentTransform();
            ResetAgentsTransform();



            //From static, move to self
            foreach (var item in team)
                item.script.behavior = BehaviorType.Self;

            currentEpisode++;
            OnEpisodeBegin();
        }
        protected virtual void OnEpisodeBegin()
        {
            ///<summary>
            /// Is called at the end of ResetEpisode() and at the begining of SetupEnvPositions()
            /// 
            /// </summary>
        }
        protected virtual void OnEpisodeEnd(ref AI ai)
        {
            ///<summary>
            ///Is called at the beggining of ResetEpisode()
            /// </summary>
        }
        //----------------------------------------------ENVIRONMENTAL----------------------------------------//
        private void EnvironmentSetup()
        {
            if (environmentType == TrainingEnvironment.NotSpecified)
                return;
            try
            {
                Environments = GameObject.FindGameObjectsWithTag("Environment");
            }
            catch { Debug.LogError("Environment tag doesn't exist. Please create a tag called Environment and assign to an environment!"); environmentType = TrainingEnvironment.NotSpecified; return; }

            environmentsNumber = Environments.Length;

            environmentsInitialTransform = new List<PosAndRot>[environmentsNumber];
            agentsInitialTransform = new List<PosAndRot>[environmentsNumber];
            for (int i = 0; i < environmentsNumber; i++)
            {
                environmentsInitialTransform[i] = new List<PosAndRot>();
                agentsInitialTransform[i] = new List<PosAndRot>();
            }

            if (environmentsNumber == 0)
            {
                Debug.Log("There is no Environment found. Make sure your environments have Environment tag");
                return;
            }
            if (environmentsNumber == 1)
            {
                //GetEnvironment transform
                GetAllTransforms(Environments[0].transform, ref environmentsInitialTransform[0]);

                //GetStart transform
                UnityEngine.Transform Start = Environments[0].transform.Find("Start");
                if (Start == null)//If the monoenvironment doesn't have a start, take as start the AIModel
                    GetAllTransforms(AIModel.transform, ref agentsInitialTransform[0]);
                else
                {
                    GetAllTransforms(Start, ref agentsInitialTransform[0]);
                    Start.gameObject.SetActive(false);
                }
            }
            else
                for (int i = 0; i < environmentsNumber; i++)
                {
                    GetAllTransforms(Environments[i].transform, ref environmentsInitialTransform[i]);
                    List<PosAndRot> Start = new List<PosAndRot>();
                    UnityEngine.Transform StartTransform = Environments[i].transform.Find("Start");
                    GetAllTransforms(StartTransform, ref Start);
                    StartTransform.gameObject.SetActive(false);
                    agentsInitialTransform[i] = Start;
                }




            if (environmentType == TrainingEnvironment.MultipleLayersMultipleEnvironments)
                episodesPerEvolution = environmentsNumber;
            if (environmentType == TrainingEnvironment.SingleLayerMonoEnvironment)
                teamSize = environmentsNumber;

        }
        private void NextEnvironment()
        {
            //ALWAYS CALL RESET ENVIRONMENT TRANSFORM, THEN RESET AGENT TRANSFORM
            if (environmentType == TrainingEnvironment.MultipleLayersMultipleEnvironments)
            {
                currentEnvironment++;
                if (currentEnvironment == environmentsNumber)
                    currentEnvironment = 0;
            }
        }
        private void ResetEnvironmentTransform()
        {
            if (environmentType == TrainingEnvironment.NotSpecified)
                return;
            //SingleLayer Environment - reset only current env
            if (environmentType == TrainingEnvironment.MultipleLayersMonoEnvironment || environmentType == TrainingEnvironment.MultipleLayersMultipleEnvironments)
                ApplyAllTransforms(ref Environments[currentEnvironment], in environmentsInitialTransform[currentEnvironment]);
            //MultiLayer Environment - reset all environments
            else
                for (int i = 0; i < environmentsNumber; i++)
                    ApplyAllTransforms(ref Environments[i], in environmentsInitialTransform[i]);
        }
        private void ResetAgentsTransform()
        {
            if (environmentType == TrainingEnvironment.NotSpecified)
                for (int i = 0; i < team.Length; i++)
                    team[i].script.ResetToInitialPosition();
            else if (environmentType == TrainingEnvironment.MultipleLayersMonoEnvironment || environmentType == TrainingEnvironment.MultipleLayersMultipleEnvironments)
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
                ConvertColorToColor32(color, ref color32);
                statColor = GetRichTextColorFromColor32(color32);
            }
            statData.Append("<b>|Episode: <color=" + statColor + ">");
            statData.Append(currentEpisode);
            statData.Append("</color>\n");

            statData.Append("<b>|Generation: ");
            statData.Append(currentEpisode / episodesPerEvolution);
            statData.Append("\n");
            {//Colorize
                Color tlcolor = Color.Lerp(Color.red, Color.green, timeLeft / maxTimePerEpisode);
                Color32 tlcolor32 = new Color32();
                ConvertColorToColor32(tlcolor, ref tlcolor32);
                statColor = GetRichTextColorFromColor32(tlcolor32);
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
                    ConvertColorToColor32(color, ref color32);
                    StringBuilder colorString = new StringBuilder();
                    colorString.Append("#");
                    colorString.Append(GetHexFrom(color32.r));
                    colorString.Append(GetHexFrom(color32.g));
                    colorString.Append(GetHexFrom(color32.b));

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
            if (Graph == null)
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

                    xPos = zeroX + (i + 1) * xUnit;          //step
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

                    xPos = zeroX + (i + 1) * xUnit;          //step
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
                statistic.Append("\n                    Evolution - NO  | This Gen MaxFitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This Gen MaxFitness: ");
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
                statistic.Append("\n                    Evolution - NO  | This Gen MaxFitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This Gen MaxFitness: ");
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
                statistic.Append("\n                    Evolution - NO  | This Gen MaxFitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.modelNet.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This Gen MaxFitness: ");
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

        //---------------------------------------------NETWORK CHANGES---------------------------------//
        void NetworkChanges()
        {
            NeuralNetwork.mutation = mutationStrategy;
            NeuralNetwork.activation = activationType;
            NeuralNetwork.outputActivation = outputActivationType;
            NeuralNetwork.initialization = initializationType;
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
        private static void Swap<T>(ref T[] objArray, int index1, int index2)
        {
            if (objArray == null && objArray.Length <= index1 && objArray.Length <= index2) return;

            var temp = objArray[index1];
            objArray[index1] = objArray[index2];
            objArray[index2] = temp;
        }
        private static void ConvertColorToColor32(Color color, ref Color32 color32)
        {
            color32.r = System.Convert.ToByte(color.r * 255f);
            color32.g = System.Convert.ToByte(color.g * 255f);
            color32.b = System.Convert.ToByte(color.b * 255f);
            color32.a = System.Convert.ToByte(color.a * 255f);
        }
        private static string GetRichTextColorFromColor32(Color32 color)
        {
            string clr = "#";
            clr += GetHexFrom(color.r);
            clr += GetHexFrom(color.g);
            clr += GetHexFrom(color.b);
            return clr;
        }
        private static string GetHexFrom(int value)
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
        private static string GetHexDigFromIntDig(int value)
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
        private static void ReverseString(ref string str)
        {
            char[] charArray = str.ToCharArray();
            System.Array.Reverse(charArray);
            str = new string(charArray);
        }
        private void ResetFitEverywhere()
        {
            for (int i = 0; i < team.Length; i++)
            {
                team[i].script.SetFitnessTo(0f);
                team[i].fitness = 0f;
            }
        }
        private void UpdateFitnessInArray()
        {
            //Update fitness in team array
            for (int i = 0; i < team.Length; i++)
                team[i].fitness = team[i].script.GetFitness();
        }
        //-------------------------------------------------BUTTONS------------------------------------------//
        void SaveBrains()
        {
            saveBrains = false;

            string saveDir = Application.streamingAssetsPath + "/Saves/";
            if (!Directory.Exists(saveDir))
                Directory.CreateDirectory(saveDir);

            int howMany = (int)((float)team.Length - Mathf.Sqrt(team.Length));
            SortTeam();
            for (int i = team.Length - 1; i >= howMany; i--)
            {
                string path = team[i].script.GetPath();
                NeuralNetwork net = new NeuralNetwork(team[i].script.network);//Here was made a copy due to some weird write access error
                NeuralNetwork.WriteBrain(net, path);
            }

        }
    }

    public struct AI
    {
        public GameObject agent;
        public Agent script;
        public float fitness;
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

        //Accesible to user
        public void AddObservation(float observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        public void AddObservation(int observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full.");
                return;
            }
            buffer[sizeIndex++] = observation1;
        }
        public void AddObservation(uint observation1)
        {
            if (sizeIndex == buffer.Length)
            {
                Debug.Log("SensorBuffer is full.");
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
    }
    public struct ActionBuffer
    {
        private float[] action;
        public ActionBuffer(float[] actions)
        {
            action = actions;
        }
        public ActionBuffer(int capacity)
        {
            action = new float[capacity];
        }
        public float[] GetBuffer()
        {
            return action;
        }
        public int GetBufferCapacity()
        {
            if (action == null)
                return 0;
            else return action.Length;
        }

        public float GetAction(uint index)
        {
            try
            {
                return action[index];
            }
            catch { Debug.LogError("Action index out of range."); }
            return 0;
        }
        public void SetAction(uint index, float action1)
        {
            action[index] = action1;
        }
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
    public enum TrainingEnvironment
    {
        [Tooltip("@static environment\n@single environment\n@multiple agents")]
        NotSpecified,

        //Agents overlap eachother, environmental things are common
        [Tooltip("@single environment\n@multiple agents\n@first object with Environment tag")]
        MultipleLayersMonoEnvironment,//First found Environment is taken
        [Tooltip("@multiple environments\n@multiple agents per environment")]
        MultipleLayersMultipleEnvironments,

        //Agents work separately, environmental things are personal for each agent, layers are just positionated far away from eachother
        [Tooltip("@single environment\nsingle agent per environment")]
        SingleLayerMonoEnvironment,


        /*[Tooltip("Multiple Environments for a single agent")]
        SingleLayerMultipleEnvironments,*/
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
        ReLU,
        [Tooltip("@output: (-inf*,+inf)\n" +
                 "@best for hidden layers - low computation")]
        LeakyReLU,
        [Tooltip("@output: [-0.278, +inf)\n" +
                 "@smooth ReLU - higher computation")]
        SiLU,

    }
    public enum InitializationFunctionType
    {
        [Tooltip("@value: [0, 1]")]
        RandomValue,
        [Tooltip("@value: close normal distribution\n" +
            "@l = 0.15915f\n" +
            "@k = 2f\n" +
            "@z = 0.3373f")]
        StandardNormal1,
    }
    public enum LossFunctionType
    {
        Quadratic,
        Absolute,
        Binary
    }
}