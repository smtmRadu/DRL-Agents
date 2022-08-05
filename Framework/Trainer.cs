using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Trainer : TrainerAgent
{
    [Header("=== Environment Settings ===")]

    ///The following variables are EXAMPLES, you can keep the track of your environment as you wish
    /// Assign objects to this Serialized List            <var name="environmentObjects"<var>                <list type="GameObject"></list>
    /// This is used to keep track of initial positions   <list name="environmentObjectsPositions"<list>     <list type="Transform"></list>
    List<GameObject> environmentObjects = new List<GameObject>();
    List<Transform> environmentObjectsPositions = new List<Transform>();


    protected override void Awake()
    {
        base.Awake();
        startingPositions = new Vector3[teamSize];

        ///Initialize <var name="startingPositions"></var> elements. They are used when Episode resets. 
        for (int i = 0; i < startingPositions.Length; i++)
        {
            //EXAMPLE: 
            startingPositions[i] = Vector3.zero;
            //NECCESARRY +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        }

        //We keep the starting position of every object in the Training Environment, so keep this for as it is
        for (int i = 0; i < environmentObjects.Count; i++)
        {
            environmentObjectsPositions.Add(environmentObjects[i].transform);
        }

    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
        //You can assign different colors to your Agents 
        //EXAMPLE:
        //foreach (AI item in teamArray)
        //{
        //   item.agent.GetComponent<SpriteRenderer>().color = new Color(Random.value,Random.value, Random.value);
        //}
        //Agents are kept in an array called teamArray of type AI (check it in TrainerAgentScript if you want);
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
        for (int i = 0; i < environmentObjects.Count; i++)
        {
            environmentObjects[i].transform.position = environmentObjectsPositions[i].position;
            environmentObjects[i].transform.rotation = environmentObjectsPositions[i].rotation;
            environmentObjects[i].transform.localScale = environmentObjectsPositions[i].localScale;
        }
        //You can make changes of different things after each Episode

    }



}
