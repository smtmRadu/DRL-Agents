using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/*THINGS TO DO
 * Make a setting in brain creation, to customize the number of neurons per layer
 * 
 * Synthetize data to upload on Github
 * 
 * 
 * 
 * 
 */

public class Trainer : TrainerAgent
{
    [Header("=== Environment Settings ===")]
    public GameObject PushWall = null;
    Vector3 pushWallStartingPos;
    public float pushWallSpeed = 2f;
    protected override void Awake()
    {
        base.Awake();
        startingPositions = new Vector3[teamSize];
        for (int i = 0; i < startingPositions.Length; i++)
        {
            startingPositions[i] = Vector3.zero;
        }
        pushWallStartingPos = PushWall.transform.position;
    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
        foreach (AI member in teamArray)
        {
            member.agent.GetComponent<SpriteRenderer>().color = new Color(Random.value, Random.value, Random.value);
        }
    }
    protected override void EnvironmentAction()
    {
        base.EnvironmentAction();

        PushWall.transform.position += Vector3.right * pushWallSpeed * Time.deltaTime * 100;
    }
    protected override void ResetEpisode()
    {
        base.ResetEpisode();
        PushWall.transform.position = pushWallStartingPos;
    }



}
