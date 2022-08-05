using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIController : Agent
{
    [Space, Header("===== AI Properties =====")]
    ///Create variables like speed, jumpPower, rotation etc.
    /// Also aditional variables to control components like RigidBody or SpriteRenderer
    /// EXAMPLE:
    float speed = 1f;



    protected override void Awake()
    {
        base.Awake();

        //Awake or Set your variables if neccesary
    }
    protected override void Update()
    {
        base.Update();
        if (behaviour == BehaviourType.Heuristic)
            Heuristic();

        //Do whatever you want here

    }

    protected override void Heuristic()
    {
        ///Implement Keyboard Input for your player.
        ///Heuristic() is called in Update(), so use Time.deltaTime in your moving functions
        ///EXAMPLE: 
        ///      if(Input.GetKey(KeyCode.A))
        ///         rb.AddForce(Vector2.left * speed * Time.deltaTime)
        ///      else if(Input.GetKey(KeyCode.D))
        ///         rb.AddForce(Vector2.right * speed * Time.deltaTime)

    }
    protected override void UpdateInputs(out float[] SensorBuffer)
    {

        base.UpdateInputs(out SensorBuffer);

        //SensorBuffer has a length of it's neural network inputs
        //Insert semnificative inputs for your AI if you want good results
        
        ///EXAMPLE: Turn ON ErrorPause in UnityConsole
        ///try
        ///{
        ///     SensorBuffer[0] = transform.position.x
        ///     SensorBuffer[1] = transform.position.y
        ///     SensorBuffer[2] = target.transform.position.x
        ///     SensorBuffer[3] = target.transform.position.y
        ///     SensorBuffer[4] = You can assign a value of 1 if a RaycastHit != null
        ///     SensorBuffer[5] = RaycastHit collision collider object ID (or some specific value) to know how to behave when he meets it
        ///     SensorBuffer[6] = level Index ( if you are planning to train your AI's for different Environments
        ///}
        ///catch(System.Exception exception) 
        ///{
        ///    Debug.LogError(exception);
        ///}
    }
    protected override bool OnOutputsReceived(float[] ActionBuffer)
    {
        if (!base.OnOutputsReceived(ActionBuffer))
            return false;
        //ActionBuffer has a length of it's neural outputs with values between [0f,1f]
        //You can use the outputs to make actions similar to Heuristic
        //OnOuputsReveived() is called in Update(), so use Time.deltaTime in your moving functions
        ///EXAMPLE
        //////     OUTPUT 1:
        ///         if(ActionBuffer[0] < .5f)
        ///             rb.AddForce(Vector2.left * speed * Time.deltaTime)
        ///         else 
        ///             rb.AddForce(Vector2.right * speed * Time.deltaTime)
        ///           
        //////     OUTPUT 2:
        ///        if(ActionBuffer[1] > .5f)
        ///             Jump();


        //Do not modify this return
        return true;
    }

    public void OnCollisionEnter2D(Collision2D collision)
    {
        //Do other stuff here if you need

        if (behaviour != BehaviourType.Learning || network == null)
            return;

        ///Mostly your Agents will Get Reward and Finnish an Episode by colliding with objects

        ///EXAMPLE
        ///     if(collision.collider.CompareTag("Coin"))
        ///             AddReward(1f);
        ///     if(collision.collider.CompareTag("Trap"))
        ///             EndEpisode();
        ///     if(collision.collider.ComopareTag("FinnishLine")
        ///     {
        ///             SetReward(100f);
        ///             EndEpisode();
        ///     }        

    }
}
