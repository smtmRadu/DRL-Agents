using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.IO;
using UnityEngine.UI;
using System.IO;
using TMPro;
using UnityEditor;

/// <summary>
/// 0,0,0,0,0,0,0,0
/// .5,.5,.5,.5,.5,.5,.5,.5
/// BUG: IN UpdateBestBrain nu copiaza BestNeuralNetwork in path-ul membrului
/// </summary>

public class TrainerAgent: MonoBehaviour
{
    public GameObject AIModel;
    [Tooltip("Insert the path of the brain")]public string BrainModel;
    private string bestBrainModel = "Assets/StreamingAssets/Best_Neural_Network/BestNeuralNetwork.txt";
    private float bestFitness;
    [Space]
    [Range(1,50)]public int teamSize = 1;
    [Range(1, 1000)] public int maxStep = 10;
    public bool clearAllNNAtSessionEnd = true;
    private int currentStep = 1;
    protected List<GameObject> team = new List<GameObject>();

    Vector3 startingPos;
    bool canTrain = true;
    bool environmentCanGo=false;    


    protected virtual void Awake()
    {
        startingPos = Vector3.zero;
        
    }
    private void Start()
    {
        CreateBestNeuralNetworkIfDoesNotExist();
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
        if(AIModel.GetComponent<PlayerController>() == null)
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
            PlayerController memberScript = member.GetComponent<PlayerController>();
            memberScript.behaviour = BehaviourType.Learning;
            memberScript.CreateNeuralNetwork(true, memberScript.GetLayersFormat(), brainModelContents);
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
        UpdateBestBrain();
        foreach (var member in team)
        {
            PlayerController script = member.GetComponent<PlayerController>();
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
                if (member.GetComponent<PlayerController>().behaviour == BehaviourType.Learning)
                    return false;
            }
            catch { }
           
        }
        return true;

    }
    protected virtual void ResetEpisode()
    { 
        environmentCanGo=false;
        UpdateBestBrain();

        foreach (var member in team)
        {
            member.transform.position = startingPos;
            member.GetComponent<PlayerController>().behaviour = BehaviourType.Learning;
        }
        
        environmentCanGo = true;
        //---Environment Reset must be overridden
    }

    private void UpdateBestBrain()
    {
        //Find Best AI and it's fitness
        float bestFitInThisGen = float.MinValue;
        int bestAiIndex = -1;
        for (int i = 0; i < team.Count; i++)
        {
            float fit = team[i].GetComponent<PlayerController>().currentNNFitness;
           
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
                 File.Copy(team[bestAiIndex].GetComponent<PlayerController>().GetCurrentNetworkPath(), bestBrainModel, true);
            } catch { Debug.Log("Couldn't Update best brain"); }  
        }

        //------COPY BEST BRAIN IN ALL AI's BRAINS
        foreach (GameObject ai in team)
        {
            PlayerController script = ai.GetComponent<PlayerController>();
            script.CopyNetworkFrom(bestBrainModel);
            script.MutateHisBrain();
            script.UpdateTextFile();
        }


        //Eroare----> Cand incepe sa fie mai prost, nu i se copiaza inapoi bestBrainModelul, in schimb mai face modificari la acelasi ala vechi




    }
}
