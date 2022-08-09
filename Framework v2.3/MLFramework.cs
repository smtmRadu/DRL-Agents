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
        static MutationStrategy mutationStrategy = MutationStrategy.Strategy1;
        protected int[] layers;
        protected float[][] neurons;
        protected float[][][] weights;
        protected float fitness;
        static private float bias = 0;

        public NeuralNetwork(int[] layers)
        {
            InitializeLayers(layers);
            InitializeNeurons();
            InitializeWeights(false);
            fitness = 0f;

        }
        public NeuralNetwork(NeuralNetwork copyNN)
        {
            InitializeLayers(copyNN.layers);
            InitializeNeurons();
            InitializeWeights(false);
            CopyWeights(copyNN.weights);
        }
        public NeuralNetwork(string path)
        {
            if (new FileInfo(path).Length == 0)
            {
                Debug.LogError("The training cannot start! Reason: Brain Model uploaded file is empty");
                return;
            }
            List<string> fileLines = File.ReadAllLines(path).ToList();
            //Instatiate Neural Network
            string[] line1 = fileLines[0].Split(',');
            int[] line1_32 = new int[line1.Length];//THIS LINE REPREZENTS THE LAYERS []
            ConvertStrArrToIntArr(line1, ref line1_32);
            InitializeLayers(line1_32);
            InitializeNeurons();
            InitializeWeights(true);

            //Copy weights data
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < fileLines.Count - 1; i++)//First and last line are ignored ( first line is layerFormat, last line is Fitness)
            {
                //One line here are the weights on a single layer
                List<float[]> weightsOnLayer = new List<float[]>();

                string[] line = fileLines[i].Split(',');
                float[] line_32 = new float[line.Length];
                ConvertStrArrToFloatArr(line, ref line_32);


                //This array must be devided depeding on the previous layer number of neurons
                int numNeurOnPrevLayer = line1_32[i - 1];
                float[] weightsOnNeuron = new float[numNeurOnPrevLayer];
                int count = 0;

                for (int j = 0; j < line_32.Length; j++)
                {

                    ///Problema cand ajunge la line32.lenght -1, ultima adaugare nu o baga in weights on Layer
                    if (count < numNeurOnPrevLayer)
                    {
                        weightsOnNeuron[count] = line_32[j];
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
            this.SetWeights(weightsList.ToArray());//Final set
            float fit = new float();
            fit = float.Parse(fileLines[fileLines.Count - 1]);
            this.SetFitness(fit);

        }
        private void CopyWeights(float[][][] copyNNweights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        this.weights[i][j][k] = copyNNweights[i][j][k];
                    }
                }
            }
        }

        //-----------------INITIALIZARION-----------------//
        protected void InitializeLayers(int[] layers)
        {
            this.layers = new int[layers.Length];
            for (int i = 0; i < layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }
        }
        protected void InitializeNeurons()
        {
            List<float[]> neuronsList = new List<float[]>();
            foreach (int layer in layers)
            {
                neuronsList.Add(new float[layer]);
            }
            neurons = neuronsList.ToArray();
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
                    if(!empty)
                        SetWeightsOfASingleNeuronWithNormalDistribution(neuronWeights);
                    weightsOnLayerList.Add(neuronWeights);
                }
                weightsList.Add(weightsOnLayerList.ToArray());
            }
            weights = weightsList.ToArray();
        }
        protected void SetWeightsOfASingleNeuronWithNormalDistribution(float[] axons)
        {
            for (int i = 0; i < axons.Length; i++)
            {
                axons[i] = GetNumberFromStandardNormal();
            }
        }

        //-------------------Propagation--------------------//
        public float[] ForwardPropagation(float[] inputs)
        {
            SetInputs(inputs);

            for (int l = 1; l < layers.Length; l++)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    float value = bias;
                    for (int p = 0; p < neurons[l - 1].Length; p++)
                    {
                        value += weights[l - 1][n][p] * neurons[l - 1][p];
                    }
                    neurons[l][n] = SigmoidFunction(value);
                }
            }



            return neurons[neurons.Length - 1]; //Return the last layer (OUTPUT)
        }
        public void BackPropagation()
        {

        }

        //--------------------MUTATIONS---------------------//
        public void MutateWeights()
        {
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    for (int k = 0; k < weights[i][j].Length; k++)
                        MutateWeight(ref weights[i][j][k]);
        }
        protected void MutateWeight(ref float weight)
        {
            if (mutationStrategy == MutationStrategy.Strategy1)
                MutateByStrategy1(ref weight);
            else if (mutationStrategy == MutationStrategy.Strategy2)
                MutateByStrategy2(ref weight);


        }
        void MutateByStrategy1(ref float weight)
        {
            float randNum = UnityEngine.Random.Range(0f, 10f);

            if (randNum <= 2f)//20% chance of flip sign of the weight
            {
                weight *= -1f;
            }
            else if (randNum <= 4f)//20% chance of fully randomize weight
            {
                weight = UnityEngine.Random.Range(-.5f, .5f);
            }
            else if (randNum <= 6f)//20% chance of increase to 100 - 200 %
            {
                float factor = UnityEngine.Random.Range(0f, 1f) + 1f;
                weight *= factor;
            }
            else if (randNum <= 8f)//20% chance of decrease from 0 - 100 %
            {
                float factor = UnityEngine.Random.Range(0f, 1f);
                weight *= factor;
            }
            else
            {
            }//20% chance of NO MUTATION

        }
        void MutateByStrategy2(ref float weight)
        {
            //In this Strategy the weight is mutated by a value from 0->.5f; ///SMALL MUTATION
            float randNum = UnityEngine.Random.Range(-.5f, .5f);
            weight += randNum;
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
        static float SigmoidFunction(float value)
        {
            // Function is x = 1/(1 + e^(-x))
            return (float)1f / (1f + Mathf.Exp(-value));
        }
        static float NormalDistributionOf(float x, float sigma = 1f, float mu = 0f)
        {
            return (float)(1 / System.Math.Sqrt((2f * System.Math.PI * Mathf.Pow(sigma, 2f)))
                               * Mathf.Exp(-1f / 2f * Mathf.Pow((x - mu) / sigma, 2f)));
        }
        static float GetNumberFromStandardNormal(float l = 0.15915f, float k = 2f, float z = 0.3373f)
        {
            float x = Random.value;
            float sign = Random.value;
            if (sign > .5f)
                return (float)Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);
            else
                return (float)-Mathf.Pow(-Mathf.Log(2f * l * Mathf.PI * Mathf.Pow(x, 2f)) * z, 1f / k);


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


        //--------------Setters & Getters------------------//
        public int[] GetLayers()
        {
            return layers;
        }
        public float[][][] GetWeights()
        {
            return weights;
        }
        public void SetLayers(int[] layers)
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }
        }
        public void SetWeights(float[][][] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                for (int j = 0; j < weights[i].Length; j++)
                {
                    for (int k = 0; k < weights[i][j].Length; k++)
                    {
                        this.weights[i][j][k] = weights[i][j][k];
                    }
                }
            }
        }
    }
    public class AgentBase: UnityEngine.MonoBehaviour
    {
        [Header("===== Agent Properties =====")]
        public BehaviourType behaviour = BehaviourType.Static;
        [SerializeField, Range(1, 50), Tooltip("The number of Inputs that the Agent will receive [-1,1]")] private int spaceSize = 2;
        [SerializeField, Range(1, 15), Tooltip("The number of Outputs that the Agent will return [ 0,1]")] private int actionSize = 2;
        public bool SaveBrain = false;
        public NeuralNetwork network = null;

        [Space, Header("===== Network Properties =====")]
        [Tooltip("If path != null -> Has a model assigned + if(networkStatus) -> The model is also loaded")] public string path = null;
        [SerializeField, Range(1, 100), Tooltip("The number of Hidden Layers")] private int deep = 5;
        [SerializeField, Range(1, 100), Tooltip("The number of Neurons per Hidden Layer." + "[Usually you can set it as (spaceSize + actionSize)]")] private int neuronsPerLayer = 5;
       
        protected virtual void Update()
        {
            //SmallChecks
            BUTTONSaveBrain();
            if (behaviour == BehaviourType.Self || behaviour == BehaviourType.Learning)
                Action();
            else if (behaviour == BehaviourType.Manual)
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
                ReadBrain(path);
            }
            float[] inputs = new float[network.GetLayers()[0]];

            CollectObservations(ref inputs);
            float[] outputs = network.ForwardPropagation(inputs);
            NormalizeOutputs(ref outputs);
            OnActionReceived(in outputs);

        }
        void NormalizeOutputs(ref float[] outp)
        {
            //Outputs are in [0f,1f] Range so we move them to [-1f,1f]
            for (int i = 0; i < outp.Length; i++)
            {
                outp[i] -= .5f;
                outp[i] *= 2;
            }
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

        protected void AddReward(float reward)
        {
            if (behaviour == BehaviourType.Manual)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot add reward because neural network is null");
                return;
            }
            if (behaviour == BehaviourType.Learning)
                network.AddFitness(reward);
        }
        protected void SetReward(float reward)
        {
            if (behaviour == BehaviourType.Manual)
                return;
            if (network == null)
            {
                Debug.LogError("Cannot set reward because neural network is null");
                return;
            }
            if (behaviour == BehaviourType.Learning)
                network.SetFitness(reward);
        }
        protected void EndAction()
        {
            behaviour = BehaviourType.Static;
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

        //---------------------------------------------BUTTONS---------------------------------------------------//
        void BUTTONSaveBrain()
        {
            if(SaveBrain == true)
            {
                SaveBrain = false;
                if (network != null)
                    WriteBrain();
                else
                {
                    //CreateBrain and Write it
                    List<int> lay = new List<int>();
                    lay.Add(spaceSize);
                    for (int i = 0; i < deep; i++)
                        lay.Add(neuronsPerLayer);
                    lay.Add(actionSize);
                    this.network = new NeuralNetwork(lay.ToArray());
                    WriteBrain();
                }
                
            }
        }

        //---------------------------------------READ AND WRITE BRAIN------------------------------------------//
        public void WriteBrain(string path = null)
        {
            //1 Create File and path if not
            if (path == null)
            {
                path = Application.streamingAssetsPath + "/Neural_Networks/";
                if (!Directory.Exists(path))
                    Directory.CreateDirectory(path);

                path += "NeuralNetworkSaveID" + this.gameObject.GetInstanceID() * -1f + ".txt";
            }
            File.WriteAllText(path, string.Empty);
            File.AppendAllText(path, string.Join(",", network.GetLayers()));
            File.AppendAllText(path, "\n");

            float[][][] weights = network.GetWeights();
            foreach (float[][] layWeights in weights)
            {
                for (int i = 0; i < layWeights.Length; i++)
                {
                    File.AppendAllText(path, string.Join(",", layWeights[i]));
                    if (i < layWeights.Length - 1)
                        File.AppendAllText(path, ",");
                }
                File.AppendAllText(path, "\n");
            }
            File.AppendAllText(path, network.GetFitness().ToString());
        }
        void ReadBrain(string path)
        {
            //Reads contents of a file and setup it's own network based on these contents
            this.network = new NeuralNetwork(path);
        }

    }
    public class TrainerBase : UnityEngine.MonoBehaviour
    {
        [Header("=== Models ===")]
        public GameObject AIModel;
        [Tooltip("Insert the path of the brain")] public string brainModelPath;
        private NeuralNetwork modelNet;

        [Space(20)]
        public GameObject Environment;
        public TMPro.TMP_Text statisticsDisplay = null;
        [Tooltip("Turning this ON may decrease performance")]public bool sortedDisplay = false;
        [Tooltip("Saves all networks of the team in /Saves/ folder")public bool SaveBrains = false;

        [Space, Header("=== Training Settings ===")]
        [Range(2, 300)] public int teamSize = 5;//IT cannot be 1 because there cannot be reproduction
        [Range(1, 50), Tooltip("Episodes needed to run until passing to the next Generation")] public int episodesPerEvolution = 1;
        [Range(1, 1000), Tooltip("Total Episodes in this Training Session")] public int maxEpisodes = 100;
        [Range(1, 1000), Tooltip("Maximum time allowed per Episode")] public float maxTimePerEpisode = 100f; float timeLeft;
        public TrainingStrategy trainingStrategy = TrainingStrategy.Best;

        private int currentStep = 1;
        protected AI[] team;
        protected Transform[] agentsStartingPositions;
        protected Transform[] environmentStartingPosition;
        bool startTraining = true;


        protected virtual void Awake()
        {
            CreateDir();
            timeLeft = maxTimePerEpisode;
        }
        protected virtual void Start()
        {
            BestStrategySet();
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
            BUTTONSaveBrains();
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
        void BestStrategySet()
        {
            if (trainingStrategy == TrainingStrategy.Best)
                trainingStrategy = TrainingStrategy.Strategy1;
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
                team[i].script = member.GetComponent<Agent>() as Agent;
                team[i].fitness = 0f;
            }

            //Initialize AI
            for (int i = 0; i < team.Length; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(brainModelPath);
                script.SetFitnessTo(0f);
                script.network.MutateWeights();
                script.behaviour = BehaviourType.Learning;

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
            UpdateDisplay();
        }
        void SetupStartingPositions()
        {
            OnEpisodeBegin();
            if(agentsStartingPositions == null)
            {
                agentsStartingPositions = new Transform[team.Length];
                for (int i = 0; i < agentsStartingPositions.Length; i++)
                {
                    agentsStartingPositions[i] = AIModel.transform;
                }
            }
            if(Environment == null)
            {
                environmentStartingPosition = new Transform[0];
            }
            environmentStartingPosition = new Transform[Environment.transform.childCount];
            for (int i = 0; i < environmentStartingPosition.Length; i++)
            {
                environmentStartingPosition[i] = Environment.transform.GetChild(i).transform;
            }
        }

        //--------------------------------------------TRAINING PROCESS--------------------------------------//
        void Train()
        {
            timeLeft -= Time.deltaTime;
            EnvironmentAction();

            //Update fitness in team array
            for (int i = 0; i < team.Length; i++)
                team[i].fitness = team[i].script.GetFitness();

            if (AreAllDead() || timeLeft <= 0)
                ResetEpisode();

            if (currentStep >= maxEpisodes)
            {
                Debug.Log("Training Session Ended!");
                ResetEpisode();
                foreach (var item in team)
                    item.script.behaviour = BehaviourType.Static;
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
                if (item.script.behaviour == BehaviourType.Learning)
                    return false;
            return true;
        }
        protected void ResetEpisode()
        {
            timeLeft = maxTimePerEpisode;
            //Next Gen
            if (currentStep % episodesPerEvolution == 0)
            {
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
            for (int i = 0; i < Environment.transform.childCount; i++)
            {
                var child = Environment.transform.GetChild(i);
                child.transform.position = environmentStartingPosition[i].position;
                child.transform.rotation = environmentStartingPosition[i].rotation;
                child.transform.localScale = environmentStartingPosition[i].localScale;
            }

            //From static, move to learning
            foreach (var item in team)
                item.script.behaviour = BehaviourType.Learning;

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
        void UpdateDisplay()
        {
            //Update is called after every EpisodeReset
            if (statisticsDisplay == null)
                return;
            if (sortedDisplay == true)
                SortTeam();

            StringBuilder statData = new StringBuilder();
            statData.AppendLine("<b>Step: " + currentStep + "\n | Goal: " + modelNet.GetFitness() + "</b>");
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

            statisticsDisplay.text = statData.ToString();
        }
        //--------------------------------------------TRAINING STRATEGY----------------------------------//
        void NextGenStrategy1()
        {
            /// <summary>
            /// Half worst AI's are replaced with the a single copy of half best AI's, only the copy is mutated
            /// </summary>
            SortTeam();//The sort was made in ascending order
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
                team[team.Length - 1].script.WriteBrain(brainModelPath);
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
                    script.network.MutateWeights();
                }
            else
                for (int i = 0; i <= halfCount; i++)
                {
                    var script = team[i].script;
                    script.network = new NeuralNetwork(team[i + halfCount].script.network);
                    script.network.MutateWeights();
                }
        }
        void NextGenStrategy2()
        {

        }
        void NextGenStrategy3()
        {

        }
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
        //-------------------------------------------------BUTTONS------------------------------------------//
        void BUTTONSaveBrains()
        {
            //They will be saved in a Saves Directory descending by their fitness, so the best brain will be first
            if (SaveBrains == false)
                return;
            SaveBrains = false;

            string saveDir = Application.streamingAssetsPath + "/Saves/";
            if (!Directory.Exists(saveDir)) 
                Directory.CreateDirectory(saveDir);

            for (int i = team.Length-1; i >= 0; i--)
            {
                StringBuilder sbPath = new StringBuilder();
                sbPath.Append(saveDir);
                sbPath.Append("NeuralNetworkSaveID");
                sbPath.Append(team[i].agent.GetInstanceID() * -1f);
                sbPath.Append("(Time:");
                sbPath.Append(Time.time);
                sbPath.Append(").txt");

                string path = sbPath.ToString();
                //Make a copy of it's network
                NeuralNetwork net = new NeuralNetwork(team[i].script.network);
                try
                {
                    File.WriteAllText(path, string.Empty);
                    File.AppendAllText(path, string.Join(",", net.GetLayers()));
                    File.AppendAllText(path, "\n");

                    float[][][] weights = net.GetWeights();
                    foreach (float[][] layWeights in weights)
                    {
                        for (int j = 0; j < layWeights.Length; j++)
                        {
                            File.AppendAllText(path, string.Join(",", layWeights[i]));
                            if (j < layWeights.Length - 1)
                                File.AppendAllText(path, ",");
                        }
                        File.AppendAllText(path, "\n");
                    }
                    File.AppendAllText(path, net.GetFitness().ToString());
                }
                catch { }
                //Only some of them can be saved
            }

        }
    }




    public struct AI
    {
        public GameObject agent;
        public Agent script;
        public float fitness;
    }
    public enum MutationStrategy
    {
        Strategy1,
        Strategy2
    }
    public enum BehaviourType
    {
        Static,
        Manual,
        Self,
        Learning
    }
    public enum TrainingStrategy
    {
        //The call for Strategy is in ResetEpisode Method
        [Tooltip("Chooses the best Strategy according to the developer")]
        Best,
        [Tooltip("Only Best AI Reproduce + All NextGen get Mutated")]
        Strategy1,
        [Tooltip("Half Best AI Reproduce + Only copies get Mutated")]
        Strategy2,
        [Tooltip("20% get Best Brain | 80% Half Best AI Reproduce + Only copies get Mutated")]
        Strategy3,

    }
}