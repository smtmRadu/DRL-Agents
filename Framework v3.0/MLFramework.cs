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

        //------------------------ACTIVATION FUNCTIONS-----=-------------------//
        static float Activate(float value, bool outputLayer = false)
        {
            if (outputLayer == false)
            {
                if (activation == ActivationFunctionType.Sigmoid)
                    return ActivationFunctionSigmoidLogistic(value);
                else if (activation == ActivationFunctionType.Tanh)
                    return ActivationFunctionHyperbolicTangent(value);
                else if (activation == ActivationFunctionType.ReLU)
                    return ActivationFunctionRectifiedLinearUnit(value);
                else if (activation == ActivationFunctionType.SoftPlus)
                    return ActivationFunctionSoftPlus(value);
                else if (activation == ActivationFunctionType.BinaryStep)
                    return ActivationFunctionBinaryStep(value);
            }
            else
            {
                if (outputActivation == ActivationFunctionType.Sigmoid)
                    return ActivationFunctionSigmoidLogistic(value);
                else if (outputActivation == ActivationFunctionType.Tanh)
                    return ActivationFunctionHyperbolicTangent(value);
                else if (outputActivation == ActivationFunctionType.ReLU)
                    return ActivationFunctionRectifiedLinearUnit(value);
                else if (outputActivation == ActivationFunctionType.SoftPlus)
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
        static float ActivationFunctionSigmoidLogistic(float value)
        {
            //values range [0,1]
            // Function is x = 1/(1 + e^(-x))
            return (float)1f / (1f + Mathf.Exp(-value));
        }
        static float ActivationFunctionHyperbolicTangent(float value)
        {
            return (float)System.Math.Tanh((double)value);
        }
        static float ActivationFunctionRectifiedLinearUnit(float value)
        {
            return Mathf.Max(0, value);
        }
        static float ActivationFunctionSoftPlus(float value)
        {
            return Mathf.Log(1 + Mathf.Exp(value));
        }

        //-----------------------INITIALIZATION FUNCTIONS----------------------//
        static float InitializationFunctionStandardNormal(float l = 0.15915f, float k = 2f, float z = 0.3373f)
        {
            float x = Random.value;
            float sign = Random.value;
            if (sign > .5f)
                return (float)Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);
            else
                return (float)-Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);


        }
        static float InitializationFunctionRandomvalue()
        {
            return (float)Random.value;
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
    }
    public class AgentBase : UnityEngine.MonoBehaviour
    {
        [Header("===== Agent Properties =====")]
        public BehaviorType behavior = BehaviorType.Static;
        [Tooltip("Assign a path to a brain model")] public string path = null;
        [Tooltip("Press this button to create a new brain using Network Properties settings or to save his current brain.\nCheck StreamingAssets/Neural_Networks.")] public bool SaveBrain = false;
        public NeuralNetwork network = null;

        [Space, Header("===== Network Properties =====")]
        [SerializeField, Range(1, 50), Tooltip("The number of Inputs that the Agent will receive [-1,1]")] private int spaceSize = 2;
        [SerializeField, Range(1, 15), Tooltip("The number of Outputs that the Agent will return [-1,1]")] private int actionSize = 2;
        [SerializeField, Tooltip("Each element is a hidden layer\nEach value is the number of neurons\n@biases not count")] private List<uint> hiddenLayers;

        /* [SerializeField, Range(1, 100), Tooltip("The number of Hidden Layers")] private int deep = 5;
         [SerializeField, Range(1, 100), Tooltip("The number of Neurons per Hidden Layer")] private int neuronsPerLayer = 5;*/

        protected virtual void Update()
        {
            //SmallChecks
            BUTTONSaveBrain();
            if (behavior == BehaviorType.Self)
                Action();
            else if (behavior == BehaviorType.Manual)
                Manual();

        }

        void Action()
        {
            if (network == null) // case he self controls
            {
                if (path == null || new FileInfo(path).Length == 0)
                {
                    Debug.LogError("Cannot Self Control because the Brain Path uploaded is invalid");
                    return;
                }
                this.network = new NeuralNetwork(path);
            }
            float[] inputs = new float[network.GetLayers()[0]];
            CollectObservations(ref inputs);
            float[] outputs = network.ForwardPropagation(inputs);
            OnActionReceived(in outputs);

        }
        //-------------------------------------------FOR USE WHEN OVERRIDING----------------------------------------------//
        protected virtual void Manual()
        {

        }
        protected virtual void CollectObservations(ref float[] SensorBuffer)
        {

        }
        protected virtual void OnActionReceived(in float[] ActionBuffer)
        {

        }

        public void AddReward(float reward, bool evenIfEndedAction = false)
        {
            if (behavior == BehaviorType.Manual)
                return;
            if (evenIfEndedAction == false && behavior == BehaviorType.Static)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot add reward because neural network is null");
                return;
            }
            network.AddFitness(reward);
        }
        public void SetReward(float reward, bool evenIfEndedAction = false)
        {
            if (behavior == BehaviorType.Manual)
                return;
            if (evenIfEndedAction == false && behavior == BehaviorType.Static)
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
            behavior = BehaviorType.Static;
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
            pathsb.Append(this.gameObject.GetInstanceID());
            pathsb.Append(".txt");


            return pathsb.ToString();
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
                    lay.Add(spaceSize);

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

    }
    public class TrainerBase : UnityEngine.MonoBehaviour
    {
        [Header("=== Models ===")]
        public GameObject AIModel;
        [Tooltip("Brain model used to start the training with")] public string brainModelPath;
        private NeuralNetwork modelNet;
        [Tooltip("Used to reset dynamic environmental objects's positions")] public GameObject Environment;

        [Space(20), Tooltip("Save networks of best Ai's in /Saves/ folder before moving to the nextGen\nNumber of saves = Sqrt(Team Size)")] public bool saveBrains = false;

        [Space(20), Header("=== Statistics Display ===")]
        [Tooltip("Load a Canvas TMPro to watch the current performance of AI's")] public TMPro.TMP_Text Labels = null;
        [Tooltip("Load a Canvas RectTransform to watch a Gizmos graph in SceneEditor")] public RectTransform Graph = null;
        List<float> bestResults;//memorize best results for every episode
        List<float> averageResults;//memorize avg results for every episode

        [Space, Header("=== Training Settings ===")]
        [Range(3, 500)] public int teamSize = 5;//IT cannot be 1 or 2, otherwise strategies will not work
        [Range(1, 10), Tooltip("Episodes needed to run until passing to the next Generation")] public int episodesPerEvolution = 1;
        [Range(1, 1000), Tooltip("Total Episodes in this Training Session")] public int maxEpisodes = 100;
        [Range(1, 100), Tooltip("Maximum time allowed per Episode")] public float maxTimePerEpisode = 100f; float timeLeft;

        [Tooltip("-In the beggining use Strategy1.\n-When you see your AI's performance slows down, switch to Strategy2.\n-To finetune your last Brain, use Strategy3.")]
        public TrainingStrategy trainingStrategy = TrainingStrategy.Strategy1;
        public MutationStrategy mutationStrategy = MutationStrategy.Classic;
        public ActivationFunctionType activationType = ActivationFunctionType.Tanh;
        public ActivationFunctionType outputActivationType = ActivationFunctionType.Tanh;
        public InitializationFunctionType initializationType = InitializationFunctionType.StandardNormal1;



        private int currentStep = 1;
        protected AI[] team;
        protected Transform[] agentsStartingPositions;
        protected EnvObject[] environmentObjects;
        bool startTraining = true;


        protected virtual void Awake()
        {
            CreateDir();
            timeLeft = maxTimePerEpisode;
            bestResults = new List<float>();
            averageResults = new List<float>();
        }
        protected virtual void Start()
        {
            if (!TrainingPreparation())
            {
                startTraining = false;
                return;
            }
            SetupTeam();
            SetupStartingPositions();

        }
        protected virtual void Update()
        {
            NetworkChanges();
            if (startTraining)
                Train();

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

            //Colorize AI's if possible
            foreach (var item in team)
            {
                if (item.agent.TryGetComponent<SpriteRenderer>(out var spriteRenderer))
                {
                    if (spriteRenderer == null)
                        break;
                    spriteRenderer.color = new Color(Random.value, Random.value, Random.value);
                }
            }

            //Turn Off the model
            AIModel.SetActive(false);
            UpdateDisplay();
        }
        void SetupStartingPositions()
        {
            OnEpisodeBegin();
            if (agentsStartingPositions == null)
            {
                agentsStartingPositions = new Transform[team.Length];
                for (int i = 0; i < agentsStartingPositions.Length; i++)
                {
                    agentsStartingPositions[i] = AIModel.transform;
                }
            }
            if (Environment != null)
            {
                environmentObjects = new EnvObject[Environment.transform.childCount];
                for (int i = 0; i < environmentObjects.Length; i++)
                {
                    environmentObjects[i].obj = Environment.transform.GetChild(i).gameObject;
                    environmentObjects[i].startingTransform = new GameObject().transform;
                    environmentObjects[i].startingTransform.position = environmentObjects[i].obj.transform.position;
                    environmentObjects[i].startingTransform.rotation = environmentObjects[i].obj.transform.rotation;
                    environmentObjects[i].startingTransform.localScale = environmentObjects[i].obj.transform.localScale;
                }
            }
        }

        //--------------------------------------------TRAINING PROCESS--------------------------------------//
        void Train()
        {
            timeLeft -= Time.deltaTime;
            EnvironmentAction();

            UpdateFitnessInArray();

            if (AreAllDead() || timeLeft <= 0)
                ResetEpisode();

            if (currentStep >= maxEpisodes)
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

            //Graph Related
            bestResults.Add(team[team.Length - 1].fitness);
            averageResults.Add(FindAverageResult());

            timeLeft = maxTimePerEpisode;
            //Next Gen
            if (currentStep % episodesPerEvolution == 0)
            {
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

            //Reset AI's Position
            for (int i = 0; i < team.Length; i++)
            {
                var agent = team[i].agent;
                agent.transform.position = agentsStartingPositions[i].position;
                agent.transform.rotation = agentsStartingPositions[i].rotation;
                agent.transform.localScale = agentsStartingPositions[i].localScale;
            }

            //Reset Environment Position
            if (Environment != null)
                for (int i = 0; i < Environment.transform.childCount; i++)
                {
                    var child = Environment.transform.GetChild(i);
                    child.transform.position = environmentObjects[i].startingTransform.position;
                    child.transform.rotation = environmentObjects[i].startingTransform.rotation;
                    child.transform.localScale = environmentObjects[i].startingTransform.localScale;
                }

            //From static, move to self
            foreach (var item in team)
                item.script.behavior = BehaviorType.Self;

            currentStep++;
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
        //-----------------------------------------------STATISTICS---------------------------------------------//
        void UpdateDisplay()
        {
            //Update is called after every EpisodeReset
            if (Labels == null)
                return;

            SortTeam();

            StringBuilder statData = new StringBuilder();
            statData.Append("<b>|Epsiode: ");
            statData.Append(currentStep);
            statData.Append("\n");
            statData.Append("|Goal: ");
            statData.Append(modelNet.GetFitness());
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
                    ConvertColorToColor32(ref color, ref color32);
                    StringBuilder colorString = new StringBuilder();
                    colorString.Append("#");
                    colorString.Append(GetHexFrom(color32.r));
                    colorString.Append(GetHexFrom(color32.g));
                    colorString.Append(GetHexFrom(color32.b));

                    line.Append("<color=" + colorString.ToString() + ">");
                }
                catch (System.Exception e) { hasColor = false; Debug.Log(e); }

                line.Append("ID: ");
                line.Append(item.agent.GetInstanceID().ToString());
                //IF COLORIZED
                if (hasColor)
                    line.Append("</color>");

                line.Append(" | Fitness: ");
                line.Append(item.script.GetFitness());

                line.Append("\n");
                statData.AppendLine(line.ToString());
            }

            Labels.text = statData.ToString();
        }
        private void OnDrawGizmos()
        {
            //Draw Graph
            try
            {

                if (Graph == null)
                    return;

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



                float xUnit = xSize / currentStep;
                /* if (bestResults.Count > 0)
                     if (goal < bestResults[bestResults.Count - 1])
                         goal = bestResults[bestResults.Count - 1];*/

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
                        Gizmos.DrawSphere(neuronsPosition[i][j], scale * 6000f);

                //Draw Biases
                Gizmos.color = biasColor;
                for (int i = 0; i < biasesPosition.Length; i++)
                {
                    Gizmos.DrawSphere(biasesPosition[i], scale * 6000f);
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
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentStep);
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
            //GetHalfBestBrains and assign to worst guys
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
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentStep);
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
            //BUILD STATISTIC
            StringBuilder statistic = new StringBuilder();
            statistic.Append("Step: ");
            statistic.Append(currentStep);

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
            //STACK OVERFLOW PROBLEMS
            return null;
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
            else return arr;
        }

        //------------------------------------------COMPLEMENTARY METHODS-----------------------------------//
        private static void Swap<T>(ref T[] objArray, int index1, int index2)
        {
            if (objArray == null && objArray.Length <= index1 && objArray.Length <= index2) return;

            var temp = objArray[index1];
            objArray[index1] = objArray[index2];
            objArray[index2] = temp;
        }
        private static void ConvertColorToColor32(ref Color color, ref Color32 color32)
        {
            color32.r = System.Convert.ToByte(color.r * 255f);
            color32.g = System.Convert.ToByte(color.g * 255f);
            color32.b = System.Convert.ToByte(color.b * 255f);
            color32.a = System.Convert.ToByte(color.a * 255f);
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

    public struct EnvObject
    {
        public GameObject obj;
        public Transform startingTransform;
    }
    public struct AI
    {
        public GameObject agent;
        public Agent script;
        public float fitness;
    }

    public enum BehaviorType
    {
        [Tooltip("Cannot move at all")]
        Static,
        [Tooltip("Can move only by user input")]
        Manual,
        [Tooltip("Moves independently")]
        Self

    }
    public enum TrainingStrategy
    {
        [Tooltip("Half best AI reproduce + Only copies get mutated")]
        Strategy1,
        [Tooltip("(1/3) Worst AI get best brain + Mutation | (2/3) 50% Half best AI reproduce + Only copies get mutated")]
        Strategy2,
        [Tooltip("Only best AI reproduce + All copies get mutated")]
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
        [Tooltip("@output: 0 or 1")]
        BinaryStep,
        [Tooltip("@output: (0, 1)")]
        Sigmoid,
        [Tooltip("@output: (-1, 1)")]
        Tanh,
        [Tooltip("@output: [0, +inf)")]
        ReLU,
        [Tooltip("@output: [0, +inf)" +
                 "@smooth ReLU)]")]
        SoftPlus,

    }
    public enum InitializationFunctionType
    {
        [Tooltip("@value: [0, 1]")]
        RandomValue,
        [Tooltip("@value: close normal distribution" +
            "@l = 0.15915f" +
            "@k = 2f" +
            "@z = 0.3373f")]
        StandardNormal1,

    }
}