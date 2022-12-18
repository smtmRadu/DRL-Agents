using DRLAgents;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Versioning;
using UnityEditor;
using UnityEngine;
namespace DRLAgents
{
    [DisallowMultipleComponent]
    public class Heuristics : MonoBehaviour
    {
        public HeuristicModule mode = HeuristicModule.Collect;
        [Tooltip("@file that holds the training data used for heuristic training")] public TextAsset trainingDataFile;
        [Tooltip("@reset the environment transforms when agent action ended")] public GameObject Environment;
        [Tooltip("@watch the progression of the error")] public RectTransform errorGraph;

        [Range(1, 1000), Tooltip("@number of parsings through the training data.")] public uint epochs = 100;
        [Range(0, 300), Tooltip("@data collection time in seconds.\n@33.33 samples per second")] public float sessionLength = 60;
        [Tooltip("@collect samples when no user inputs")] public bool CollectPasiveActions = false;



        [Space(10), SerializeField, ReadOnly, Tooltip("@ shows the error of a mini-batch")] public float error;


        [Header("@HyperParameters")]
        [Range(0.0001f, 1f), Tooltip("@modification strength per iteration")] public float learnRate = 0.1f;
        [Tooltip("@improves gradient descent time"), Range(0, 0.99f)] public float momentum = 0.9f;
        [Tooltip("@impact to weight decay"), Range(0, 0.1f)] public float regularization = 0.001f;
        [Tooltip("@how whole training data is splitted into mini-batches.\n@if = 1 -> Full Batch\n@else -> Mini Batch"), Range(0.01f, 1.00f)] public float batchSplit = 0.10f;
        [Tooltip("@loss function type")] public LossFunctionType lossFunction = LossFunctionType.Quadratic;
    }
    [CustomEditor(typeof(Heuristics), true), CanEditMultipleObjects]
    class ScriptlessHeuristics : Editor
    {
        private static readonly string[] _dontIncludeMe = new string[] { "m_Script" };

        public override void OnInspectorGUI()
        {
            serializedObject.Update();

            DrawPropertiesExcluding(serializedObject, _dontIncludeMe);

            serializedObject.ApplyModifiedProperties();
        }
    }
}