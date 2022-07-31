using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.IO;
using UnityEngine.UI;
using System.IO;
using TMPro;
using UnityEditor;


//This TrainerAgent class works like this
/*Takes all AI's and the best brain 
 * every evolutionStep* times, the brain with the highest fitness is copyied in BestNeuralNetwork
 * the nextGeneration get the best brain that is now and a mutation is applied for each of them
 * the procces repeats
 */
//The improved TrainerAgent class works like this
/*Takes all AI's and give them this brain
 * half of them are mutated
 * every evolutionStep* times, the best half brains are saved in a Directory called Best_Neural_Networks  [also the best brain from all of them is saved in Best_Neural_Network]
 * foreach good brain -> on brain is assigned to two AI's, second AI is mutated
 * the process repeats
 */

public class TrainerAgent: MonoBehaviour
{
    [Header("=== AI Models ===")]
    public GameObject AIModel;
    [Tooltip("Insert the path of the brain")]public string BrainModel;
    [Space(20)]
    //Some things must be modified in order to efficiently use it:
    // 1. in EndEpisode() -> UpdateTextFile only if this is enabled
    // 2. on ResetEpisodeTo(0,!fastTraining) calls -> Do not UpdateTextFile because it doesn t matter(the network var is compared at the end)
    //[SerializeField, Tooltip("Enabling this option the trainer will not Overwrite AI's File after each Episode")] bool fastTraining = true;
    public bool clearAllNNAtSessionEnd = true;
    public bool mutateFromStart = true;
    private string bestBrainModel = "Assets/StreamingAssets/Best_Neural_Network/BestNeuralNetwork.txt";
    private float bestFitness;
    [Space, Header("=== Team Settings ===")]
    [Range(1,100)]public int teamSize = 1;
    [Range(1, 20),Tooltip("Frequency for calling NextGeneration")] public int evolutionStep = 5;
    [Range(1, 1000),Tooltip("Total Episodes in this Training Session")] public int maxStep = 10;
    
    
    private int currentStep = 1;
    protected List<GameObject> team = new List<GameObject>();

    Vector3 startingPos;
    bool canTrain = true;
    bool environmentCanGo=false;    


    protected virtual void Awake()
    {
        startingPos = Vector3.zero;
        CreateBestNeuralNetworkIfDoesNotExist();
    }
    private void Start()
    {
        
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
        if(environmentCanGo)
        {
            EnvironmentAction();
        }

    }

    //---------------------------------------------------------------------------------//
    void CheckTrainingPreparation()
    {
        if(AIModel == null)
        {
            canTrain = false;
            Debug.Log("The training cannot start! Reason: No AI Model uploaded");
            return;
        }
        if(AIModel.GetComponent<AIController>() == null)
        {
            canTrain = false;
            Debug.Log("The training cannot start! Reason: No Brain Model uploaded");
            return;
        }
    } 
    void CreateBestNeuralNetworkIfDoesNotExist()
    {
        if(!Directory.Exists(Application.streamingAssetsPath + "/Best_Neural_Network/"))
             Directory.CreateDirectory(Application.streamingAssetsPath + "/Best_Neural_Network/");
        if (!File.Exists(bestBrainModel))
             File.Create(bestBrainModel);
    }
    protected virtual void SetupTeam()
    {
        for (int i = 0; i < teamSize; i++)
        {
            GameObject member = Instantiate(AIModel, startingPos, Quaternion.identity);
            team.Add(member);
        }
        string[] brainModelContents = File.ReadAllLines(BrainModel);
        if(brainModelContents.Length == 0)
        {
            Debug.Log("Brain Model file is Empty! Please Insert a Valid Model!");
            canTrain = false;
            return;
        }
        foreach (GameObject member in team)
        {
            AIController memberScript = member.GetComponent<AIController>();
            memberScript.CreateNeuralNetwork(true, memberScript.GetLayersFormat(), brainModelContents);

            memberScript.ResetFitnessTo(0f, true);

            if(mutateFromStart)
                memberScript.MutateHisBrain();
            
           
            memberScript.behaviour = BehaviourType.Learning;
            member.transform.position = startingPos;
        }
        bestFitness = float.Parse(brainModelContents[brainModelContents.Length - 1]);
    }
 
    
    //------------------------------------TRAINING PROCESS-----------------------------//
    IEnumerator Training()
    {
        if(!environmentCanGo)
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
        NextGeneration();
        foreach (var member in team)
        {
            AIController script = member.GetComponent<AIController>();
            script.behaviour = BehaviourType.Static;
            File.Delete(script.path);
        }


    }
    protected virtual void EnvironmentAction()
    {

    }
    private bool AreAllDead()
    {
        foreach (var member in team)
        {
            try
            {
                if (member.GetComponent<AIController>().behaviour == BehaviourType.Learning)
                    return false;
            }
            catch { }
           
        }
        return true;

    }
    protected virtual void ResetEpisode()
    { 
        environmentCanGo=false;

        if(currentStep%evolutionStep == 0)
             NextGeneration();

        foreach (var member in team)
        {
            member.transform.position = startingPos;
            member.GetComponent<AIController>().behaviour = BehaviourType.Learning;
        }
        
        environmentCanGo = true;
        //---Environment Reset must be overridden
    }
    private void NextGeneration()
    {
        //Find Best AI and it's fitness
        float bestFitInThisGen = float.MinValue;
        int bestAiIndex = -1;
        for (int i = 0; i < team.Count; i++)
        {
            float fit = team[i].GetComponent<AIController>().currentNNFitness;
           
            if (fit > bestFitInThisGen)
            {
                bestFitInThisGen = fit;
                bestAiIndex = i;
            }
        }
        //In case the new generation is weaker than the previous one | their brain evoluted in a wrong way  ===> do not update the best brain
        if (bestFitInThisGen < this.bestFitness)
            Debug.Log("Step: " + currentStep + " | NextGen - NO  | This Gen MaxFitness: " + bestFitInThisGen + " < " + this.bestFitness);
        else
        {
            Debug.Log("Step: " + currentStep + " | NextGen - YES | This Gen MaxFitness: " + bestFitInThisGen + " > " + this.bestFitness);
            this.bestFitness = bestFitInThisGen;

            //-----COPY THIS STEP BRAIN TO BEST BRAIN
            try
            {
                 File.Copy(team[bestAiIndex].GetComponent<AIController>().GetCurrentNetworkPath(), bestBrainModel, true);
            } catch { Debug.Log("Couldn't Update best brain"); }  
        }

        //------COPY BEST BRAIN IN ALL AI's BRAINS AND RESET THE FITNESS TO 0 --> the data in saveFile remains with the old fitness
        foreach (GameObject ai in team)
        {
            AIController script = ai.GetComponent<AIController>();
            script.CopyNetworkFrom(bestBrainModel);
            script.SetNetworkFromFile(script.path, ref script.network);
            script.ResetFitnessTo(0f, true);
            script.MutateHisBrain();
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
}
