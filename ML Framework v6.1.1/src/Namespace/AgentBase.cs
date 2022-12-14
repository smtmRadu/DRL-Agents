using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace MLFramework
{
    public class AgentBase : MonoBehaviour
    {
        [Header("@Status")]
        public BehaviorType behavior = BehaviorType.Static;
        [Tooltip("@model used for this agent")] public TextAsset networkModel = null;
        [Tooltip("@if has brain: saves current brain data\n@else: creates a brain using Network Properties\n@folder: StreamingAssets/Neural_Networks. \n@default naming format or uses Path")] public bool SaveNetwork = false;
        public NeuralNetwork network = null;
        List<PosAndRot> initialPosition = new List<PosAndRot>(); static int parseCounter = 0;

        [Header("@Network Layout")]
        [SerializeField, Range(1, 50), Tooltip("The number of Inputs that the Agent will receive")] private int sensorSize = 2;
        [SerializeField, Range(1, 20), Tooltip("The number of Outputs that the Agent will return")] private int actionSize = 2;
        [SerializeField, Tooltip("Each element is a hidden layer\nEach value is the number of neurons\n@biases not count")] private List<uint> hiddenLayers;

        [Header("@Network Functions")]
        [Tooltip("@activation function used in hidden layers")]
        public ActivationFunctionType activationType = ActivationFunctionType.Tanh;
        [Tooltip("@activation function used for output layer\n@influences the actionBuffer values")]
        public ActivationFunctionType outputActivationType = ActivationFunctionType.Tanh;
        [Tooltip("@initializes weights and biases of a newly created network")]
        public InitializationFunctionType initializationType = InitializationFunctionType.StandardDistribution;


        private AgentHeuristics Heur;



        //ONLY HEURISTIC 
        private List<Sample> samplesCollected;//used only when appending/learning
        private List<List<Sample>> batches;//used only when learning
        int totalTrainingSamples = 0;
        //Environmental
        private List<PosAndRot> environmentInitialTransform;
        //ErrorStatistic
        bool callStatistic = false;
        float maxErrorFound = 0;
        uint startingEpochs = 0;
        List<Vector2> errorPoints = new List<Vector2>();

        //Mini batch help vars only
        int miniBatchSize;
        int miniBatchesNumber;
        int currentMiniBatchIndex;//used for data spliting in minibatches and also when training 1 batch at a time

        //Sampling collection
        int frameIndex;//frame index while collecting data

        protected virtual void Awake()
        {
            Heur = GetComponent<AgentHeuristics>();

            GetAllTransforms();
            //On heuristic and self the brains are made directly in CollectHeuristicData and SelfAction,this may cause in action lag

            startingEpochs = Heur.epochs;

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
                if (networkModel == null)
                {
                    Debug.LogError("<color=red>Cannot Self Control because network model is missing.</color>");
                    return;
                }
                this.network = new NeuralNetwork(networkModel.text);

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
            if (Heur.module == HeuristicModule.Learn)
                ProcessOneBatch();
            else
                CollectTrainingData();
        }

        //-------------------------------------------HEURISTIC TRAINING--------------------------------------------------//
        void CollectTrainingData()
        {
            Heur.sessionLength -= Time.deltaTime;
            if (Heur.sessionLength <= 0)
            {
                behavior = BehaviorType.Static;

                Debug.Log("<color=#64de18>Appending <color=#e405fc>" + samplesCollected.Count + " </color>training samples...</color>");

                
                File.AppendAllLines(
                                Heur.trainingDataFile != null ?
                                AssetDatabase.GetAssetPath(Heur.trainingDataFile) :
                                (GetHeuristicSamplesPath() + "TrainingData_" + UnityEngine.Random.Range(0, 1000).ToString() + ".txt"),
                                GetLinesFromBatchList()
                                );
                AssetDatabase.Refresh();

                Debug.Log("<color=#64de18>Data collected succesfully.</color>");

                return;
            }

            //Get inputs
            SensorBuffer inputs = new SensorBuffer(network.GetInputsNumber());
            CollectObservations(ref inputs);
            //Get userOutputs
            ActionBuffer desiredOutputs = new ActionBuffer(network.GetOutputsNumber());
            Heuristic(ref desiredOutputs);
            OnActionReceived(desiredOutputs);

            // //This part is used to save a sample less than 1 per frame -> 2/3 samples from 3 frames are deprecated
            frameIndex++;
            try
            {
                if (frameIndex % (int)(0.03f / Time.deltaTime) != 0)
                    return;
                //at 600 fps, deltaTime = 0.00166 -> (0.03/deltaTime) = 18 frames
                //a sample is collected every 18 frames -> 33.33 samples per second
                //at 100 fps, deltaTime = 0.01 -> (0.03/0.01) = 3 frames
                // a sample is collected every 3 frames -> 33.33 samples per second

                //If i want to change this, also change the debug.log from heuristic preparation where ~minutes are shown

                // 0.02 -> 50 samples per second
                // 0.03 -> 33.33 samples per second
                // 0.04 -> 25 samples per second

            }
            catch { /*divided by 0*/}


            //Creating sample
            Sample sample = new Sample();
            sample.inputs = inputs.GetBuffer();
            sample.expectedOutputs = desiredOutputs.GetBuffer();
            //Check if null inputs KILLABLE
            if (Heur.CollectPasiveActions == true)
                samplesCollected.Add(sample);
            else
            {
                //collect only active actions
                foreach (var output in sample.expectedOutputs)
                    if (output != 0)
                    {
                        //active action found
                        samplesCollected.Add(sample);
                        break;
                    }
            }
        }
        void HeuristicPreparation()
        {
            HeuristicEnvironmentSetup();
            HeuristicOnSceneReset();

            if (network == null)
            {
                if (networkModel == null)
                {
                    Debug.LogError("<color=red>Brain Path is invalid</color>");
                    return;
                }
                this.network = new NeuralNetwork(networkModel.text);

                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;
                    
                HeuristicOnSceneReset();
            }

            if (behavior == BehaviorType.Manual)
                return;

            if (Heur.module == HeuristicModule.Learn)
            {
                batches = new List<List<Sample>>();

                //Debug.Log("<color=#64de18>Collecting data from file <color=grey>" + samplesPath + "</color>...</color>");
                if (Heur.trainingDataFile == null)
                {
                    Debug.Log("<color=red>TrainingData file is invalid.</color>");
                    behavior = BehaviorType.Static;
                    return;
                }

                string[] stringBatch = File.ReadAllLines(AssetDatabase.GetAssetPath(Heur.trainingDataFile));

                miniBatchesNumber = (int)(1f / Heur.batchSplit);
                miniBatchSize = stringBatch.Length / (2 * miniBatchesNumber);

                currentMiniBatchIndex = 0;

                for (int i = 0; i < stringBatch.Length / 2; i++)
                {
                    AddSampleToBatches(GetSampleFromData(stringBatch[i * 2], stringBatch[i * 2 + 1]));
                    totalTrainingSamples++;
                }
                currentMiniBatchIndex = 0;

                Debug.Log("<color=#64de18>Total samples: <color=#e405fc>" + totalTrainingSamples + "</color> = <color=#e405fc>"
                          + miniBatchesNumber + "</color> mini-batches <color=#e02810>X </color><color=#e405fc>" + miniBatchSize + "</color> samples " +
                          "(~<color=#e02810>" + totalTrainingSamples / 2000 + "</color> minutes of training data)" +
                          ". Agent is learning. Force Stop the simulation by sliding epochs to 0.</color>");
            }
            else if (Heur.module == HeuristicModule.Collect)
            {
                samplesCollected = new List<Sample>();
                frameIndex = 0;
                //case file exists
                try
                {
                    FileInfo fi = new FileInfo(AssetDatabase.GetAssetPath(Heur.trainingDataFile));
                    if (fi.Exists)
                    {
                        Debug.Log("<color=#64de18>Collecting gameplay from user...</color>");
                        return;
                    }
                }
                catch { }

                //case file do not exist
                Debug.Log("<color=64de18>Training data file created.</color>");
                Debug.Log("<color=#64de18>Collecting data from user...</color>");

            }
        }
        void AddSampleToBatches(Sample sample)
        {
            if (batches.Count == 0)
            {
                batches.Add(new List<Sample>() { sample });
                return;
            }
            if (batches[currentMiniBatchIndex].Count > miniBatchSize)
            {
                batches.Add(new List<Sample>());
                currentMiniBatchIndex++;
            }
            batches[currentMiniBatchIndex].Add(sample);

        }
        void ProcessOneBatch()
        {
            if (Heur.epochs > 0)
            {
                if (currentMiniBatchIndex > miniBatchesNumber - 1)
                { currentMiniBatchIndex = 0; Heur.epochs--; }


                network.InitMainGradientsArrays();

                //Update gradients multithreaded
                System.Threading.Tasks.Parallel.ForEach(batches[currentMiniBatchIndex], (sample) =>
                {
                    network.UpdateGradients(sample.inputs, sample.expectedOutputs, Heur.lossFunction);
                });

                //Apply gradients
                network.ApplyGradients(Heur.learnRate, batches[currentMiniBatchIndex].Count, Heur.momentum, Heur.regularization);

                Heur.error = network.GetError() / batches[currentMiniBatchIndex].Count;//is a mini batch error
                callStatistic = true;

                currentMiniBatchIndex++;
            }
            else
            {
                Debug.Log("<color=#4db8ff>Heuristic training has ended succesfully.</color><color=grey> Watch your agent current performance.</color>");
                NeuralNetwork.WriteBrain(network, networkModel);

                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;


                ResetEnvironmentToInitialPosition();
                ResetToInitialPosition();
                behavior = BehaviorType.Self;
            }
        }


        private void OnDrawGizmos()
        {
            if (Heur == null || Heur.errorGraph == null)
                return;
            //adds a point with the error everytime is called
            if (!callStatistic)
                return;
            float xSize = Heur.errorGraph.rect.width;
            float ySize = Heur.errorGraph.rect.height / 2;
            try
            {
                Gizmos.matrix = Heur.errorGraph.localToWorldMatrix;

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
                if (Heur.error > maxErrorFound)
                {
                    float oldMaxError = maxErrorFound;
                    maxErrorFound = Heur.error;
                    xUnit = xSize / (startingEpochs * miniBatchesNumber);
                    yUnit = ySize / maxErrorFound;
                    for (int i = 0; i < errorPoints.Count; i++)
                    {
                        errorPoints[i] = new Vector2(zeroX + xUnit * i, errorPoints[i].y * (oldMaxError / maxErrorFound));
                    }
                }
                else
                {
                    xUnit = xSize / (startingEpochs * miniBatchesNumber);
                    yUnit = ySize / maxErrorFound;
                }


                Vector2 newErrorPoint = new Vector2(zeroX + xUnit * errorPoints.Count, yUnit * Heur.error);
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
            }
            catch { }
            //draw network
            try
            {
                float SCALE = .05f;
                Color emptyNeuron = Color.yellow;
                Color biasColor = Color.green;
                NeuralNetwork nety = network;



                int[] layers = nety.GetLayers();
                float[][][] weights = nety.GetWeights();
                float[][] biases = nety.GetBiases();

                Vector2[][] neuronsPosition = new Vector2[layers.Length][];//starts from up-left
                Vector2[] biasesPosition = new Vector2[layers.Length - 1];//one for each layer

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
                            Gizmos.color = new Color(0, 0, weightValue);
                        else Gizmos.color = new Color(-weightValue, 0, 0);
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
                                Gizmos.color = new Color(0, 0, weightValue);
                            else
                                Gizmos.color = new Color(-weightValue, 0, 0);
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
            callStatistic = false;
        }
        private void HeuristicEnvironmentSetup()
        {
            if (Heur.Environment == null)
                return;

            environmentInitialTransform = new List<PosAndRot>();
            GetAllTransforms(Heur.Environment.transform, ref environmentInitialTransform);
        }
        private void ResetEnvironmentToInitialPosition()
        {
            if (Heur.Environment == null)
                return;
            ApplyAllTransforms(ref Heur.Environment, environmentInitialTransform);
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
            string[] lines = new string[samplesCollected.Count * 2];
            int i = 0;
            foreach (Sample sample in samplesCollected)
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
        public string GetRandomNameTXT()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("Network");
            sb.Append(((int)this.gameObject.GetInstanceID()) * (-1));
            sb.Append(".txt");
            return sb.ToString();
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
            if (SaveNetwork == false)
                return;

            SaveNetwork = false;
            if (network != null)
                NeuralNetwork.WriteBrain(network, networkModel, null);
            else
            {
                if (ZeroFoundInHiddenLayers())
                    Debug.Log("<color=red>Cannot instantiate a neural network with (hidden) layers with 0 neurons</color>");
                //Set hyperparameters
                NeuralNetwork.activation = activationType;
                NeuralNetwork.outputActivation = outputActivationType;
                NeuralNetwork.initialization = initializationType;

                //createBrain and Write it
                List<int> lay = new List<int>();
                lay.Add(sensorSize);

                if (hiddenLayers != null)
                    foreach (int neuronsNumber in hiddenLayers)
                    {
                        lay.Add(neuronsNumber);
                    }

                lay.Add(actionSize);
                this.network = new NeuralNetwork(lay.ToArray());
                NeuralNetwork.WriteBrain(network, null, GetRandomNameTXT());

            }

            bool ZeroFoundInHiddenLayers()
            {
                foreach (var item in hiddenLayers)
                {
                    if (item == 0)
                        return true;
                }
                return false;
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
}
