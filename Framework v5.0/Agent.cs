using UnityEngine;
using MLFramework;
public class Agent : AgentBase
{
    [Header("=== AI Properties ===")]
    public float speed;

    protected override void Awake()
    {
        base.Awake();
    }
    protected override void Update()
    {
        base.Update();
    }
    protected override void CollectObservations(ref SensorBuffer sensorBuffer)
    {
        //sensorBuffer.Length == sensorSize
        //sensorBuffer.AddObservation(<observation>);
    }
    protected override void OnActionReceived(in ActionBuffer actionBuffer)
    {
        //actionBuffer.Length == actionSize 
        //actionBuffer.GetAction(<index>);
    }
    protected override void Heuristic(ref ActionBuffer actionsOut)
    {
        //fill actionsOut in the reverse way of OnActionReceived()
        //actionsOut.SetAction(<index>,<action>);
    }
    protected override void HeristicOnSceneReset()
    {
        //[OPTIONAL] immediate actions after scene is reset on heuristic
        //Example: randomly move scene objects (get reference through this script)
    }

    //Use in CollisionCollider2D, CollisionTrigger2D, Update(), etc. 
    //AddReward() - cumulative reward
    //SetReward() - static reward (*not used when EpisodesPerEvolution > 1)
    //EndAction() - behaviour becomes static
}
