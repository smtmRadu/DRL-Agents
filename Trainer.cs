using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Trainer : TrainerAgent
{
    [Header("Environment")]
    public GameObject PushWall = null;
    Vector3 pushWallStartingPos;
    public float pushWallSpeed = 2f;
    protected override void Awake()
    {
        base.Awake();
        pushWallStartingPos = PushWall.transform.position;
    }
    protected override void SetupTeam()
    {
        base.SetupTeam();
        foreach (GameObject member in team)
        {
            member.GetComponent<SpriteRenderer>().color = new Color(Random.value, Random.value, Random.value);
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
