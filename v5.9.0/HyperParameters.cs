using MLFramework;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.Versioning;
using UnityEngine;
public class HyperParameters : MonoBehaviour
{
    [Header("@Network")]
    [Tooltip("@activation function used in hidden layers")]
    public ActivationFunctionType activationType = ActivationFunctionType.Tanh;
    [Tooltip("@activation function used for output layer\n@influences the actionBuffer values")]
    public ActivationFunctionType outputActivationType = ActivationFunctionType.Tanh;
    [Tooltip("@initializes weights and biases of a newly created network")]
    public InitializationFunctionType initializationType = InitializationFunctionType.StandardDistribution;

    [Header("@GradientDescent")]
    [Range(0.0001f, 1f), Tooltip("@modification strength per iteration")] public float learnRate = 0.1f;
    [Tooltip("@improves gradient descent time"), Range(0, 0.99f)] public float momentum = 0.9f;
    [Tooltip("@impact to weight decay"), Range(0, 0.1f)] public float regularization = 0.001f;
    [Tooltip("@loss function type")] public LossFunctionType lossFunction = LossFunctionType.Quadratic;
    [Tooltip("@how whole training data is splitted into mini-batches.\n@if = 1 -> Full Batch\n@else -> Mini Batch"), Range(0.01f, 1.00f)] public float batchSplit = 0.10f;

}
