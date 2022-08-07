using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLFramework;
public class Trainer : TrainerBase
{
    protected override void SetupTeam()
    {
        base.SetupTeam();
        foreach (var item in team)
        {
            if(item.agent.TryGetComponent<SpriteRenderer>(out var spriteRenderer))
            {
                spriteRenderer.color = new Color(Random.value, Random.value, Random.value);
            }
        }
    }
    protected override void EnvironmentAction()
    {
        base.EnvironmentAction();
    }
    protected override void ResetEpisode()
    {
        base.ResetEpisode();
    }
}
