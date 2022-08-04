using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AIController: Agent
{
    [Space, Header("===== AI Properties =====")]

    ///Create variables like speed, jumpPower, rotation etc.
    /// Also aditional variables to control components like RigidBody or SpriteRenderer
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

        ///EXAMPLE: 
        ///      if(Input.GetKey(KeyCode.A))
        ///         rb.AddForce(Vector2.left * speed * Time.deltaTime)
        ///      else if(Input.GetKey(KeyCode.D))
        ///         rb.AddForce(Vector2.right * speed * Time.deltaTime)

    }
    protected override void UpdateInputs(out float[] SensorBuffer)
    {

        base.UpdateInputs(out SensorBuffer);

        Vector3 start = transform.position - new Vector3(0f, 1f, 0f);
        RaycastHit2D[] hits = Physics2D.RaycastAll(start, transform.right, 6f, bitMask);
        Debug.DrawLine(start, start + new Vector3(1f, 0f, 0f) * 6, Color.red);
        SensorBuffer[0] = 0f;

        foreach (var hit in hits)
        {
            { SensorBuffer[0] = 1f; break; }
        }



        start = transform.position;
        hits = Physics2D.RaycastAll(start, new Vector3(1f, 1f, 0), 7f, bitMask);
        Debug.DrawLine(start, start + new Vector3(1f, 1f, 0) * 7, Color.blue);
        SensorBuffer[1] = 0f;
        foreach (var hit in hits)
        {
            { SensorBuffer[1] = 1f; break; }
        }

        return;
        string inputs = string.Join(" | ", SensorBuffer);
        Debug.Log(inputs);
    }
    protected override bool OnOutputsReceived(float[] ActionBuffer)
    {
        if (!base.OnOutputsReceived(ActionBuffer))
            return false;
        //ActionBuffer has a lenght of your neural outputs with values between [0f,1f]
        //You can use the outputs to make actions similar to Heuristic

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

        if (behaviour != BehaviourType.Leaning || networkStatus == null)
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
