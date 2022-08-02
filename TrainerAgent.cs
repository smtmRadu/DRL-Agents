using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.IO;
using UnityEngine.UI;
using System.IO;
using System.Text;
using TMPro;
using UnityEditor;


//This TrainerAgent class works like this
/*Takes all AI's and the best brain 
 * every evolutionStep* times, the brain with the highest fitness is copyied in BestNeuralNetwork
 * the nextGeneration get the best brain that is now and a mutation is applied for each of them
 * the procces repeats
 */
//The improved TrainerAgent class works like this -> Uses NextGeneration2
/*Takes all AI's and give them this brain
 * half of them are mutated
 * every evolutionStep* times, the best half brains are saved in a Directory called Best_Neural_Networks  [also the best brain from all of them is saved in Best_Neural_Network]
 * foreach good brain -> on brain is assigned to two AI's, second AI is mutated
 * the process repeats
 */

public class TrainerAgent : MonoBehaviour
{
    [Header("=== AI Models ===")]
    public GameObject AIModel;
    [Tooltip("Insert the path of the brain")] public string BrainModel;
    [Space(20)]
    //Some things must be modified in order to efficiently use it:
    // 1. in EndEpisode() -> UpdateTextFile only if this is enabled
    // 2. on ResetEpisodeTo(0,!fastTraining) calls -> Do not UpdateTextFile because it doesn t matter(the network var is compared at the end)
    //[SerializeField, Tooltip("Enabling this option the trainer will not Overwrite AI's File after each Episode")] bool fastTraining = true;
    [Tooltip("All files in /Neural_Networks/ will be deleted at the Start of a new Session.\n " +
    "This helps with Folder overflowing.")]public bool removeOldLogs = true;
    [Tooltip("Turning this OFF will run the first step using the unmodified BrainModel")]public bool mutateAllFromStart = true;
    private string bestBrainModel = "Assets/StreamingAssets/Best_Neural_Network/BestNeuralNetwork.txt";
    private string bestHalfBrainModelsFolder = "Assets/StreamingAssets/Best_Neural_Networks/";
    private float bestFitness;

    [Space,SerializeField] TMPro.TMP_Text statisticsDisplay = null;

    [Space, Header("=== Team Settings ===")]
    [Range(1, 100)] public int teamSize = 1;
    [Range(1, 5), Tooltip("Frequency for calling NextGeneration")] public int evolutionStep = 5;
    [Range(1, 1000), Tooltip("Total Episodes in this Training Session")] public int maxStep = 10;
    public TrainingStrategy trainingStrategy = TrainingStrategy.Best;

    private int currentStep = 1;
    protected AI[] teamArray;

    protected Vector3[] startingPositions;
    bool canTrain = true;
    bool environmentCanGo = false;


    //Task -> resolve color translation to Hexa
 
    protected virtual void Awake()
    {
        CreateDirAndBestBrainFile();
    }
    private void Start()
    {
        BestStrategySet();
        CheckTrainingPreparation();
        if (!canTrain)
            return;
        SetupTeam();


    }
    void Update()
    {
        if (canTrain == true)
        {

            canTrain = false;
            StartCoroutine(Training());
        }
        if (environmentCanGo)
        {
            EnvironmentAction();
        }
    }

    //---------------------------------------------------------------------------------//
    void CheckTrainingPreparation()
    {
        if (AIModel == null)
        {
            canTrain = false;
            Debug.Log("The training cannot start! Reason: No AI Model uploaded");
            return;
        }
        if (AIModel.GetComponent<AIController>() == null)
        {
            canTrain = false;
            Debug.Log("The training cannot start! Reason: No Brain Model uploaded");
            return;
        }
    }
    void CreateDirAndBestBrainFile()
    {
        if (!Directory.Exists(Application.streamingAssetsPath + "/Best_Neural_Network/"))
            Directory.CreateDirectory(Application.streamingAssetsPath + "/Best_Neural_Network/");
        if (!File.Exists(bestBrainModel))
            File.Create(bestBrainModel);

        //Clear Neural_Networks File
        if(removeOldLogs)
            foreach(string file in Directory.GetFiles("Assets/StreamingAssets/Neural_Networks/"))
               File.Delete(file);

        return;


       /* //Create a Directory for Best_Neural_Networks  -> Used for the Second Training Stategy
        if (!Directory.Exists(Application.streamingAssetsPath + "/Best_Neural_Networks/"))
            Directory.CreateDirectory(Application.streamingAssetsPath + "/Best_Neural_Networks/");*/
    }
    void BestStrategySet()
    {
        if (trainingStrategy == TrainingStrategy.Best)
            trainingStrategy = TrainingStrategy.Strategy3;
    }
    protected virtual void SetupTeam()
    {
        teamArray = new AI[teamSize];

        //Instantiate AI's and teamArray
        for (int i = 0; i < teamSize; i++)
        {
            GameObject member = Instantiate(AIModel, startingPositions[i], Quaternion.identity);
            teamArray[i].agent = member;
            teamArray[i].controller = member.GetComponent<AIController>() as AIController;
            teamArray[i].fitness = teamArray[i].controller.currentNNFitness;
        }

        //Check Brain Model
        string[] brainModelContents = File.ReadAllLines(BrainModel);
        if (brainModelContents.Length == 0)
        {
            Debug.Log("Brain Model file is Empty! Please Insert a Valid Model and Restart!");
            canTrain = false;
            return;
        }

        //Initialize AI's
        for (int i = 0; i < teamArray.Length; i++)
        {
            var controller = teamArray[i].controller;
            controller.CreateNeuralNetwork(true, controller.GetLayersFormat(), brainModelContents);
            controller.ResetFitnessTo(0f, true);
            if (mutateAllFromStart)
                controller.MutateHisBrain();
            controller.behaviour = BehaviourType.Learning;

            teamArray[i].agent.transform.position = startingPositions[i];


        }


        //Set best Fitness
        bestFitness = float.Parse(brainModelContents[brainModelContents.Length - 1]);

        //Set All fitnesses to 0
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].controller.ResetFitnessTo(0f, true);
            teamArray[i].fitness = teamArray[i].controller.currentNNFitness;
        }
        UpdateStatisticsDisplay();
    }


    //------------------------------------TRAINING PROCESS-----------------------------//
    IEnumerator Training()
    {
        if (!environmentCanGo)
            environmentCanGo = true;

        if (AreAllDead())
        {
            ResetEpisode();
            currentStep++;
        }


        if (currentStep >= maxStep)
        {
            StopTraining();
            yield break;
        }
        else
        {
            yield return null;
            StartCoroutine(Training());
        }
        yield return null;
    }
    void StopTraining()
    {
        environmentCanGo = false;
        Debug.Log("Training Session Ended! \nBehaviour Mode for all AIs was set to Heuristic");
        NextGeneration1();
        foreach (AI aI in teamArray)
        {
            aI.controller.behaviour = BehaviourType.Static;
            File.Delete(aI.controller.path);
        }
    }
    protected virtual void EnvironmentAction()
    {

    }
    private bool AreAllDead()
    {
        foreach (AI item in teamArray)
        {
            if(item.controller.behaviour == BehaviourType.Learning)
                return false;
        }
        return true;
        //If they are not all Static, it means they are not all dead

    }
    protected virtual void ResetEpisode()
    {
        environmentCanGo = false;
        UpdateStatisticsDisplay();
        //Update fitnesses in Array
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].fitness = teamArray[i].controller.currentNNFitness;
        }//Update fitness in array ( THIS CAN BE UPDATED JUST WHEN NEXT EVOLUTION IS HAPPENING for EFFICIENCY )

        //Next Gen
        if (currentStep % evolutionStep == 0)
        {
            switch (trainingStrategy)
            {
                case TrainingStrategy.Strategy1:
                    NextGeneration1();
                    break;
                case TrainingStrategy.Strategy2:
                    NextGeneration2();
                    break;
                case TrainingStrategy.Strategy3:
                    NextGeneration3();
                    break;
                default:
                    Debug.LogError("Training Strategy is NULL");
                    break;

            }
            //Reset fitness to 0 in next training steps
            for (int i = 0; i < teamArray.Length; i++)
            {
                teamArray[i].fitness = 0f;
            }

        }

        //Reset vars
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].agent.transform.position = startingPositions[i];
            teamArray[i].controller.behaviour = BehaviourType.Learning;
        }
    
        environmentCanGo = true;
        //---Environment Reset must be overridden
    }
    private void UpdateStatisticsDisplay()
    {
        //Update is called after every EpisodeReset
        if (statisticsDisplay == null)
            return;

        StringBuilder statData = new StringBuilder();
        statData.AppendLine("<b>Step: " + currentStep + "\n | Goal: " + bestFitness + "</b>");
        foreach (AI item in teamArray)
        {
            StringBuilder line = new StringBuilder();

            //Try COLORIZE
            bool hasColor = true;
            try
            {
                Color color = item.agent.GetComponent<SpriteRenderer>().color;

                string colorString = "#F02941";
                line.Append("<color=" + colorString + ">");
            }
            catch{ hasColor = false; }

            line.Append("ID: ");
            line.Append(item.agent.GetInstanceID().ToString());
            line.Append(" | Fitness: ");
            line.Append(item.controller.currentNNFitness);


            //IF COLORIZED
            if (hasColor)
                line.Append("</color>");
                    


            line.Append("\n");
            statData.AppendLine(line.ToString());
        }

        statisticsDisplay.text = statData.ToString();
    }

    //-------------------------------------TRAINING STRATEGIES-------------------------//
    private void NextGeneration1()
    {
        //Find Best AI and it's fitness
        float bestFitInThisGen = float.MinValue;
        int bestAiIndex = -1;
        for (int i = 0; i < teamArray.Length; i++)
        {
            float fit = teamArray[i].controller.currentNNFitness;

            if (fit > bestFitInThisGen)
            {
                bestFitInThisGen = fit;
                bestAiIndex = i;
            }
        }
        //In case the new generation is weaker than the previous one | their brain evoluted in a wrong way  ===> do not update the best brain
        if (bestFitInThisGen <= this.bestFitness)
            Debug.Log("Step: " + currentStep + " | NextGen - NO  | This Gen MaxFitness: " + bestFitInThisGen + " < " + this.bestFitness);
        else
        {
            Debug.Log("Step: " + currentStep + " | NextGen - YES | This Gen MaxFitness: " + bestFitInThisGen + " > " + this.bestFitness);
            this.bestFitness = bestFitInThisGen;

            //-----COPY THIS STEP BRAIN TO BEST BRAIN
            try
            {
                File.Copy(teamArray[bestAiIndex].controller.GetCurrentNetworkPath(), bestBrainModel, true);
            } catch { Debug.Log("Couldn't Update best brain"); }
        }

        //------COPY BEST BRAIN IN ALL AI's BRAINS AND RESET THE FITNESS TO 0 --> the data in saveFile remains with the old fitness
        foreach (AI item in teamArray)
        {
            var controller = item.controller;
            controller.CopyNetworkFrom(bestBrainModel);
            controller.SetNetworkFromFile(item.controller.path, ref item.controller.network);
            controller.ResetFitnessTo(0f, true);
            controller.MutateHisBrain();
        }
    }
    private void NextGeneration2()
    {

        SortAIsByFitness(teamArray);

        //Verification
        string str = "Step: " + currentStep + " TEAM: <color=red>";
        for (int i = 0; i < teamArray.Length; i++)
        {
            if (i == teamArray.Length / 2)
                str += " |</color><color=#4db8ff>";
            str += " | " + teamArray[i].fitness.ToString();

        }
        str += " |</color>";


        float thisGenerationBestFitness = teamArray[teamArray.Length - 1].fitness;
        if (thisGenerationBestFitness <= this.bestFitness)
            str += "\n                   | Evolution - NO  | This Gen MaxFitness: " + thisGenerationBestFitness + " < " + this.bestFitness + " |";
        else
        {
            str += "\n                   | Evolution - YES | This Gen MaxFitness: " + thisGenerationBestFitness + " > " + this.bestFitness + " |";
            this.bestFitness = thisGenerationBestFitness;
            //Try copy brain of the best AI to BestNeuralNetwork.txt
            try
            {
                File.Copy(teamArray[teamArray.Length - 1].controller.GetCurrentNetworkPath(), bestBrainModel, true);
            }
            catch { Debug.Log("Couldn't Update BestNeuralNetwork.txt"); }
        }
        Debug.Log(str);
        //Strategy3 --> Assign to first 2 Guys The Best brain and make the reproduction from the rest

        //GetHalfBestBrains and assign to worst guys
        int halfCount = teamArray.Length / 2;
        if (teamArray.Length % 2 == 0)//If Even team Size
            for (int i = 0; i < halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }
        else
            for (int i = 0; i <= halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }

        //Reset Their Fitness in the neural network and also update his file
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].controller.ResetFitnessTo(0f, true);
        }

        ///SUMMARY
        //Find best Half Ai's
        //Save their Brains in Best_Neural_Networks
        //Assign the brains two times for each AI, but mutate one of the AI's -> this way we keep a version of the old brain if the new mutated one is shit
        //Reset the Fitness to all AI's
    }
    private void NextGeneration3()
    {

        SortAIsByFitness(teamArray);

        //Verification
        string str = "";
        for (int i = 0; i < teamArray.Length; i++)
        {
            if (i == teamArray.Length / 2)
                str += "<color=red>";
            str += " | " + teamArray[i].fitness.ToString();

        }
        str += "</color>";
        Debug.Log("AfterSort: " + str + " |");

        float thisGenerationBestFitness = teamArray[teamArray.Length - 1].fitness;
        if (thisGenerationBestFitness <= this.bestFitness)
            Debug.Log("Step: " + currentStep + " | Evolution - NO  | This Gen MaxFitness: " + thisGenerationBestFitness + " < " + this.bestFitness + " |");
        else
        {
            Debug.Log("Step: " + currentStep + " | Evolution - YES | This Gen MaxFitness: " + thisGenerationBestFitness + " > " + this.bestFitness + " |");
            this.bestFitness = thisGenerationBestFitness;
            //Try copy brain of the best AI to BestNeuralNetwork.txt
            try
            {
                File.Copy(teamArray[teamArray.Length - 1].controller.GetCurrentNetworkPath(), bestBrainModel, true);
            }
            catch { Debug.Log("Couldn't Update BestNeuralNetwork.txt"); }
        }

        //Strategy3 --> Assign to first 2 Guys The Best brain and make the reproduction from the rest

        //GetHalfBestBrains and assign to worst guys
        int halfCount = teamArray.Length / 2;
        if (teamArray.Length % 2 == 0)//If Even team Size
            for (int i = 0; i < halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }
        else
            for (int i = 0; i <= halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }

        //Reset Their Fitness in the neural network and also update his file
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].controller.ResetFitnessTo(0f, true);
        }

        ///SUMMARY
        //Find best Half Ai's
        //Save their Brains in Best_Neural_Networks
        //Assign the brains two times for each AI, but mutate one of the AI's -> this way we keep a version of the old brain if the new mutated one is shit
        //Reset the Fitness to all AI's
    }

    private void SortAIsByFitness(AI[] tm)
    {
        //InsertionSort
        for (int i = 1; i < tm.Length; i++)
        {
            var key = tm[i];
            int j = i - 1;
            while(j >= 0 && tm[j].fitness > key.fitness)
            {
                tm[j + 1] = tm[j];
                j--;
            }
            tm[j + 1] = key;
        }
    }
    
    //------------------------------------Setters And Getters------------------------//
    int GetCurrentStep()
    {
        return currentStep;
    }
    float GetBestFitness()
    {
        return bestFitness;
    }

    //-----------------------------------Complementary Methods------------------------//
    private static void Swap<T>(ref T[] objArray, int index1, int index2)
    {
        if (objArray == null && objArray.Length <= index1 && objArray.Length <= index2) return;

        var temp = objArray[index1];
        objArray[index1] = objArray[index2];
        objArray[index2] = temp;
    }
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
    [Tooltip("10% get Best Brain | 90% Half Best AI Reproduce + Only copies get Mutated")]
    Strategy3,

}
public struct AI
{
    public GameObject agent;
    public AIController controller;
    public float fitness;
}