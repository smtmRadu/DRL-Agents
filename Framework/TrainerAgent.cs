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
    //IN the first Step AI's are not mutated
    [Header("=== AI Models ===")]
    public GameObject AIModel;
    [Tooltip("Insert the path of the brain")] public string BrainModel;
    [Space(20)]
    private bool fastTraining = false;
    //Some things must be modified in order to efficiently use it:
    // 1. in EndEpisode() -> UpdateTextFile only if this is enabled
    // 2. on ResetEpisodeTo(0,!fastTraining) calls -> Do not UpdateTextFile because it doesn t matter(the network var is compared at the end) HERE WAS CHANGED
    //[SerializeField, Tooltip("Enabling this option the trainer will not Overwrite AI's File after each Episode")] bool fastTraining = true;
    //INSERT MUTATION POWER -> or some sort of mutation Strategy

    [Tooltip("All files in /Neural_Networks/ will be deleted at the Start of a new Session.\n " +
    "This helps with Folder overflowing.")]public bool removeOldLogs = true;
    private string bestBrainModel = "Assets/StreamingAssets/Best_Neural_Network/BestNeuralNetwork.txt";
    private string bestHalfBrainModelsFolder = "Assets/StreamingAssets/Best_Neural_Networks/";
    private float bestFitness;

    [Space,SerializeField] TMPro.TMP_Text statisticsDisplay = null;

    [Space, Header("=== Team Settings ===")]
    [Range(2, 100)] public int teamSize = 1;//IT cannot be 1 because there cannot be reproduction
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
        if (!Directory.Exists(Application.streamingAssetsPath + "/Neural_Networks/"))
            Directory.CreateDirectory(Application.streamingAssetsPath + "/Neural_Networks/");
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
            trainingStrategy = TrainingStrategy.Strategy2;
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
            controller.ResetFitnessTo(0f, !fastTraining);
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
            try
            {
                StartCoroutine(Training());
            }
            catch
            {
                maxStep = currentStep - 1;
                Debug.LogError("Training Session has finnished prematurely due to Stack Overflow. You can stop your Training Now, and we will set your maxStep for your next session.");
            }
        }
        yield return null;//This is good to be here
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
            ResetFitnessEverywhere();
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
        for(int i = teamArray.Length-1; i >= 0; --i)
        {
            AI item = teamArray[i];
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
            catch(System.Exception e){ hasColor = false; Debug.Log(e); }

            line.Append("ID: ");
            line.Append(item.agent.GetInstanceID().ToString());
            //IF COLORIZED
            if (hasColor)
                line.Append("</color>");

            line.Append(" | Fitness: ");
            line.Append(item.controller.currentNNFitness);

            line.Append("\n");
            statData.AppendLine(line.ToString());
        }

        statisticsDisplay.text = statData.ToString();
    }

    //-------------------------------------TRAINING STRATEGIES-------------------------//
    private void NextGeneration1()
    {
        UpdateStatisticsDisplay();
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
            controller.ResetFitnessTo(0f, !fastTraining);
            controller.MutateHisBrain();
        }
    }
    private void NextGeneration2()
    {

        SortAIsByFitness(teamArray);
        UpdateStatisticsDisplay();
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
    }
    private void NextGeneration3()
    {

        SortAIsByFitness(teamArray);
        UpdateStatisticsDisplay();
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

        //Strategy3 --> 20% of stupid guys get the best brain model
        int twentyPerCent;
        if (teamArray.Length < 10)
            twentyPerCent = 1;
        else
            twentyPerCent = (int)(teamArray.Length * (float)20f / 100f);
        //twentyPerCent represents the number of member that are inside 20% 
        //if there are less than 10 members, only 1 guy will use the brain model always
        int halfCount = teamArray.Length / 2;
        if (teamArray.Length % 2 == 0)//If Even team Size
            for (int i = twentyPerCent; i < halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }
        else
            for (int i = twentyPerCent; i <= halfCount; i++)
            {
                var controller = teamArray[i].controller;
                controller.CopyNetworkFrom(teamArray[i + halfCount].controller.path);//Copy Brain From AI with his index+halfCount
                controller.SetNetworkFromFile(controller.path, ref controller.network);
                controller.MutateHisBrain();
            }
        //now copy best brain over first 20%
        for (int i = 0; i < twentyPerCent; i++)
        {
            var controller = teamArray[i].controller;
            controller.CopyNetworkFrom(bestBrainModel);
            controller.SetNetworkFromFile(controller.path, ref controller.network);
            controller.MutateHisBrain();
        }
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
    private void ResetFitnessEverywhere()
    {
        //Reset Their Fitness in the neural network and also update his file
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].controller.ResetFitnessTo(0f, !fastTraining);
        }
        //Reset fitness to 0 in next training steps
        for (int i = 0; i < teamArray.Length; i++)
        {
            teamArray[i].fitness = 0f;
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
        
        while(value > 0)
        {
            remainder = value % 16;
            value -= remainder;
            value /= 16;

            hexCode.Append(GetHexDigFromIntDig(remainder));
        }
        if(firstValue <= 15)
            hexCode.Append("0");
        if (firstValue == 0)//Case 0, we need to return 00
            hexCode.Append("0");

        string hex = hexCode.ToString();
        ReverseString(ref hex);
        return hex;
    }
    static string GetHexDigFromIntDig(int value)
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
    static void ReverseString(ref string str)
    {
        char[] charArray = str.ToCharArray();
        System.Array.Reverse(charArray);
        str = new string(charArray);
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
    [Tooltip("20% get Best Brain | 80% Half Best AI Reproduce + Only copies get Mutated")]
    Strategy3,

}
public struct AI
{
    public GameObject agent;
    public AIController controller;
    public float fitness;
}