using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DRLAgents
{
    public class Trainer : UnityEngine.MonoBehaviour
    {
        [Tooltip("Agent model gameObject used as the ai")] public GameObject agentModel;
        [Tooltip("@network model used to start the training with")] public TextAsset networkModel;
        [Tooltip("@the model used updates in the first next generation")] public bool resetFitness = true;
        [Tooltip("@save networks of best Ai's before moving to the next generation.\n@number of saves = cbrt(Team Size).\n@folder: NeuralNetworks/Team_Save#.\n@last file saved is the best AI")] public bool saveNetworks = false;

        [Space, Header("@Settings")]
        [Tooltip("@resets the dynamic environmental object's positions")] public TrainingType interactionType = TrainingType.NotSpecified;
        [Range(3, 500), Tooltip("@number of clones used\n@more clones means faster reinforcement but slow performance")] public int teamSize = 10;//IT cannot be 1 or 2, otherwise strategies will not work (if there are not 3, strategy 2 causes trouble)
        [Range(1, 10), Tooltip("Episodes needed to run until passing to the next Generation\n@TIP: divide the reward given by this number")] public int episodesPerGeneration = 1;
        [Range(1, 5000), Tooltip("Total Episodes in this Training Session")] public int maxEpisodes = 1000; private int currentEpisode = 1;
        [Range(1, 60), Tooltip("Maximum time allowed per Episode\n@don't confuse with 'per Generation'")] public float episodeLength = 25f; float timeLeft;

        [Space, Header("@HyperParameters")]
        [Tooltip("@in the beggining use Strategy1.\n@if AI's performance decreases, switch to Strategy2.\n@finetune the final network using Strategy3.")]
        public TrainingStrategy trainingStrategy = TrainingStrategy.Strategy1;
        [Tooltip("@mutates the weights and biases following certain rules")]
        public MutationStrategy mutationStrategy = MutationStrategy.Classic;

        [Space, Header("@Statistics Display")]
        [Tooltip("@finds and uses 1st ObjectOfType<Camera> \n@turn off any camera scripts")] public bool cameraFollowsBestAI = true; GameObject cam; bool isOrtographic; Vector3 perspectiveOffset;
        [Tooltip("Load a Canvas TMPro to watch the current performance of AI's")] public TMPro.TMP_Text Labels = null;
        [Tooltip("Load a Canvas RectTransform to watch a Gizmos graph in SceneEditor")] public RectTransform Graph = null;
        List<float> bestResults;//memorize best results for every episode
        List<float> averageResults;//memorize avg results for every episode




        private NeuralNetwork net;
        protected AI[] team;
        private GameObject[] Environments;

        private int currentEnvironment = 0;
        protected List<PosAndRot>[] environmentsInitialTransform;//Every item is the position of a single environment. The item is a list with positions of all environment items
        protected List<PosAndRot>[] agentsInitialTransform;//Every item is the position for a single environment. The item is the AI's initial position for environment i

        int parseCounter = 0;
        bool startTraining = true;


        protected virtual void Awake()
        {
            timeLeft = episodeLength;
            bestResults = new List<float>();
            averageResults = new List<float>();


            //Cam related
            cam = FindObjectOfType<Camera>().gameObject;
            if (cam.GetComponent<Camera>().orthographic == true)
                isOrtographic = true;
            else
            { isOrtographic = false; perspectiveOffset = cam.transform.position - agentModel.transform.position; }

            //deactivate other camera components like custom scripts
            MonoBehaviour[] comps = cam.GetComponents<MonoBehaviour>();
            foreach (var item in comps)
                item.enabled = false;
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

            for (int i = 0; i < Environments.Length; i++)//do not modify this positions
                OnEpisodeBegin(ref Environments[i]);


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
        bool TrainingPreparation()
        {
            if (agentModel == null)
            {
                Debug.LogError("The training cannot start! Reason: <color=#f27602>No AI Model uploaded</color>");
                return false;
            }
            if (agentModel.GetComponent<Agent>() == null)
            {
                Debug.LogError("The training cannot start! Reason:  <color=#f27602>AI Model does not contain Agent component</color>");
                return false;
            }
            if (networkModel == null)
            {
                Debug.LogError("The training cannot start! Reason:  <color=#f27602>Network Model not uploaded</color>");
                return false;
            }
            net = new NeuralNetwork(networkModel.text);
            if (resetFitness)
                net.SetFitness(0f);
            return true;
        }
        /// <summary>
        /// Requires base.SetupTeam() to be called in the beggining.
        /// <para>Can be overridden for pre-training setup, like coloring the agents differently.
        /// </para>
        /// <para>They are auto colored differently if they have a SpriteRenderer component.</para>
        /// </summary>
        void SetupTeam()
        {
            //Instatiate AI
            team = new AI[teamSize];
            agentModel.GetComponent<Agent>().behavior = BehaviorType.Static;
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
                    GameObject member = Instantiate(agentModel, agentModel.transform.position, agentModel.transform.rotation);
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
                script.network = new NeuralNetwork(networkModel.text);
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
            agentModel.SetActive(false);
            UpdateDisplay();
        }

        //--------------------------------------------TRAINING PROCESS--------------------------------------//
        void Train()
        {
            timeLeft -= Time.deltaTime;

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

            timeLeft = episodeLength;
            //Next Gen
            if (currentEpisode % episodesPerGeneration == 0)
            {
                //Graph Related
                bestResults.Add(team[team.Length - 1].fitness);
                averageResults.Add(FindAverageResult());

                if (saveNetworks == true)
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


            for (int i = 0; i < Environments.Length; i++)//do not modify this positions
                OnEpisodeBegin(ref Environments[i]);

        }
        /// <summary>
        /// Adds actions after episode restting. Use-case: flags activations, environment repositioning etc.
        /// <para>This method is called for each Environment separately. In Environment transform, search for the object needed and modify it.</para>
        /// </summary>
        /// <param name="Environments">Environment gameObject</param>
        protected virtual void OnEpisodeBegin(ref GameObject Environment)
        {

        }
        /// <summary>
        /// Actions before episode resetting. Use-case: post-action rewards, when the agents become static.
        /// <para>This method is called for each AI separately.</para>
        /// <para>AI parameter has 3 different fields: agent, script and fitness. All are described by hovering over them.</para>
        /// </summary>
        /// <param name="ai">The Agent</param>
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
            {
                Environments = new GameObject[1];
                return;
            }

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
                    GetAllTransforms(agentModel.transform, ref agentsInitialTransform[0]);
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
                    episodesPerGeneration *= Environments.Length;
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
        internal void GetAllTransforms(UnityEngine.Transform obj, ref List<PosAndRot> inList)
        {
            parseCounter = 1;
            inList.Add(new PosAndRot(obj.position, obj.localScale, obj.rotation));
            GetChildsTransforms(ref inList, obj);
        }
        internal void ApplyAllTransforms(ref GameObject obj, in List<PosAndRot> fromList)
        {
            parseCounter = 1;
            ApplyTransform(ref obj, fromList[0]);
            AddChildsInitialTransform(ref obj, in fromList);
        }

        internal void GetChildsTransforms(ref List<PosAndRot> list, UnityEngine.Transform obj)
        {
            foreach (UnityEngine.Transform child in obj)
            {
                PosAndRot tr = new PosAndRot(child.position, child.localScale, child.rotation);
                list.Add(new PosAndRot(child.position, child.localScale, child.rotation));
                GetChildsTransforms(ref list, child);
            }
        }
        internal void AddChildsInitialTransform(ref GameObject obj, in List<PosAndRot> list)
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
        internal void ApplyTransform(ref GameObject obj, PosAndRot trnsfrm)
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
            statData.Append((currentEpisode - 1) / episodesPerGeneration);
            statData.Append("\n");
            {//Colorize
                Color tlcolor = Color.Lerp(Color.red, Color.green, timeLeft / episodeLength);
                Color32 tlcolor32 = new Color32();
                ColorConvertor.ConvertColorToColor32(tlcolor, ref tlcolor32);
                statColor = ColorConvertor.GetRichTextColorFromColor32(tlcolor32);
            }
            statData.Append("<b>|Timeleft: <color=" + statColor + ">");
            statData.Append(timeLeft.ToString("0.000"));
            statData.Append("</color>\n");

            statData.Append("|Goal: ");
            statData.Append(net.GetFitness().ToString("0.000"));
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
            if (Graph == null || net == null)
                return;
            try
            {
                float goal = net.GetFitness();
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

                    xPos = zeroX + (i + 1) * xUnit * episodesPerGeneration; //episodesPerEvolution is added, otherwise the graph will remain to short on Xaxis
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

                    xPos = zeroX + (i + 1) * xUnit * episodesPerGeneration;    //step
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
                Color emptyNeuron = Color.yellow;
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
            if (thisGenerationBestFitness < this.net.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.net.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.net.GetFitness());
                //update ModelBrain
                net = new NeuralNetwork(team[team.Length - 1].script.network);
                NeuralNetwork.WriteBrain(in net, networkModel,null);

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
            if (thisGenerationBestFitness < this.net.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.net.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.net.GetFitness());
                //update ModelBrain
                net = new NeuralNetwork(team[team.Length - 1].script.network);
                NeuralNetwork.WriteBrain(in net, networkModel, null);
            }
            Debug.Log(statistic.ToString());

            for (int i = 0; i <= team.Length / 3; i++)
            {
                var script = team[i].script;
                script.network = new NeuralNetwork(net);
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
            if (thisGenerationBestFitness < this.net.GetFitness())
            {
                statistic.Append("\n                    Evolution - NO  | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" < ");
                statistic.Append(this.net.GetFitness());
            }
            else
            {
                statistic.Append("\n                    Evolution - YES | This generation Max Fitness: ");
                statistic.Append(thisGenerationBestFitness);
                statistic.Append(" > ");
                statistic.Append(this.net.GetFitness());
                //update ModelBrain
                net = new NeuralNetwork(team[team.Length - 1].script.network);
                NeuralNetwork.WriteBrain(in net, networkModel, null);
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
            saveNetworks = false;

            //mainDir is the main Saves directory
            //saveDir is the directory made for this specific save, it is included in the main Saves directory
            string teamSaveDirPath = "Assets/Neural_Networks/Team_Save#" + System.DateTime.Now.ToString("HH-mm-ss");
            Directory.CreateDirectory(teamSaveDirPath);
            int howMany = (int)((float)team.Length - Mathf.Pow(team.Length, 1f / 3f));

            Mathf.Pow(team.Length, 0.33f);
            for (int i = howMany; i <= team.Length - 1; i++)
            {
                string name = "/Net[" + i + "]_Fit[" + team[i].script.GetFitness().ToString("0.00") + "].txt";
                NeuralNetwork net = new NeuralNetwork(team[i].script.network);//Here was made a copy due to some weird write access error
                NeuralNetwork.WriteBrain(net, null, name, teamSaveDirPath);
            }
            string color = ColorConvertor.GetRichTextColorFromColor32(new Color32((byte)0, (byte)255, (byte)38, (byte)1));

            StringBuilder message = new StringBuilder();
            message.Append("<color=");
            message.Append(color);
            message.Append(">");
            message.Append((team.Length - howMany));
            message.Append(" neural networks have been saved in </color><i>");
            message.Append(teamSaveDirPath);
            message.Append("</i>");
            Debug.Log(message.ToString());
        }
    }
    [CustomEditor(typeof(Trainer), true), CanEditMultipleObjects]
    class ScriptlessTrainerBase : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

            serializedObject.ApplyModifiedProperties();
        }
    }
}