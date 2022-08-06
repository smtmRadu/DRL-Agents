using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Trainer : TrainerBase
{
    [Header("=== Environment Settings ===")]

    ///The following variables are EXAMPLES, you can keep the track of your environment as you wish
    /// <var name="environmentObjects"<var>                       
    /// <var name="environmentObjectsInitialPositions"</var>    
    [SerializeField] GameObject[] environmentObjects;
    [SerializeField] Transform[] environmentObjectsInitialPositions;

    protected override void Awake()
    {
        base.Awake();

        ///Initialize <var name="startingPositions"></var> elements. They are used when Episode resets. 
        for (int i = 0; i < startingPositions.Length; i++)
        {
            //EXAMPLE: 
            startingPositions[i] = Vector3.zero;
        }

        //Here's automatic assign, you can assign them manually too
        //EXAMPLE:
        try
        {
            environmentObjects = GameObject.FindGameObjectsWithTag("Environment");
            environmentObjectsInitialPositions = new Transform[environmentObjects.Length];
            for (int i = 0; i < environmentObjects.Length; i++)
                environmentObjectsInitialPositions[i] = environmentObjects[i].transform;
        }
        catch { }

    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
        //Assign different colors to your Agents 
        //EXAMPLE:
        try
        {
            foreach (AI item in teamArray)
            {
                item.agent.GetComponent<SpriteRenderer>().color = new Color(Random.value, Random.value, Random.value);
            }
        }
        catch { }
        
    }
    protected override void EnvironmentAction()
    {
        base.EnvironmentAction();

        //Apply actions for each Environment Object
    }
    protected override void ResetEpisode()
    {
        //AI's positions are already reseted in base.ResetEpisode()
        base.ResetEpisode();
        //Reset environment Position
        for (int i = 0; i < environmentObjects.Length; i++)
        {
            environmentObjects[i].transform.position = environmentObjectsInitialPositions[i].position;
            environmentObjects[i].transform.rotation = environmentObjectsInitialPositions[i].rotation;
            environmentObjects[i].transform.localScale = environmentObjectsInitialPositions[i].localScale;
        }
        //Make changes of different things after each Episode

    }



}
