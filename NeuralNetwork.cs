using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NeuralNetwork
{
    protected int[] layers;
    protected float[][] neurons;
    protected float[][][] weights;
    protected float fitness;
    static private float bias = 0;
    public NeuralNetwork(int[] layers)
    {
        InitializeLayers(layers);
        InitializeNeurons();
        InitializeWeights();
        fitness = 0f;

    }
    public NeuralNetwork(NeuralNetwork copyNN)
    {
        InitializeLayers(copyNN.layers);
        InitializeNeurons();
        InitializeWeights();
        CopyWeights(copyNN.weights);
    }
    private void CopyWeights(float[][][] copyNNweights)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].Length; j++)
            {
                for (int k = 0; k < weights[i][j].Length; k++)
                {
                    this.weights[i][j][k] = copyNNweights[i][j][k];
                }
            }
        }
    }

    //-----------------INITIALIZARION-----------------//
    protected void InitializeLayers(int[] layers)
    {
        this.layers = new int[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            this.layers[i] = layers[i];
        }
    }
    protected void InitializeNeurons()
    {
        List<float[]> neuronsList = new List<float[]>();
        foreach (int layer in layers)
        {
            neuronsList.Add(new float[layer]);  
        }
        neurons = neuronsList.ToArray();    
    }
    protected void InitializeWeights()
    {
        List<float[][]> weightsList = new List<float[][]>();
        for (int i = 1; i < layers.Length; i++)
        {
            List<float[]> weightsOnLayerList = new List<float[]>();
            int neuronsInPreviousLayer = layers[i - 1];

            for (int n = 0; n < neurons[i].Length; n++)
            {
                float[] neuronWeights = new float[neuronsInPreviousLayer];
                SetWeightsOfASingleNeuronWithNormalDistribution(neuronWeights);
                weightsOnLayerList.Add(neuronWeights);
            }
            weightsList.Add(weightsOnLayerList.ToArray());
        }
        weights = weightsList.ToArray();
    }
    protected void SetWeightsOfASingleNeuronWithNormalDistribution(float[] axons)
    {
        for (int i = 0; i < axons.Length; i++)
        {
            axons[i] = GetRandomNumberFromNormalDistribution();
        }
    }

    //-------------------Propagation--------------------//
    public float[] ForwardPropagation(float[] inputs)
    {
        SetInputs(inputs);

        for (int l = 1; l < layers.Length; l++)
        {
            for (int n = 0; n < neurons[l].Length; n++)
            {
                float value = bias;
                for (int p = 0; p < neurons[l-1].Length; p++)
                {
                    value += weights[l - 1][n][p] * neurons[l - 1][p];
                }
                neurons[l][n] = SigmoidFunction(value);
            }
        }



        return neurons[neurons.Length-1]; //Return the last layer (OUTPUT)
    }
    public void BackPropagation()
    {

    }

    //--------------------MUTATIONS---------------------//
    public void MutateWeights()
    {
        for (int i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
                for (int k = 0; k < weights[i][j].Length; k++)
                    MutateWeightType1(ref weights[i][j][k]);
    }
    protected void MutateWeightType1(ref float weight)
    {
        float randNum = UnityEngine.Random.Range(0f, 10f);

        if (randNum <= 2f)//20% chance of flip sign of the weight
        {
            weight *= -1f;
        }
        else if (randNum <= 4f)//20% chance of fully randomize weight
        {
            weight = UnityEngine.Random.Range(-.5f, .5f);
        }
        else if (randNum <= 6f)//20% chance of increase to 100 - 200 %
        {
            float factor = UnityEngine.Random.Range(0f, 1f) + 1f;
            weight *= factor;
        }
        else if (randNum <= 8f)//20% chance of decrease from 0 - 100 %
        {
            float factor = UnityEngine.Random.Range(0f, 1f);
            weight *= factor;
        }
        else {
        }//20% chance of NO MUTATION




    }
    protected void MutateWeightType2(ref float weight)
    {

    }


    //---------------------FITNESS------------------------//
    public int CompareTo(NeuralNetwork other)
    {
        if (other == null) return 1;

        if (this.fitness > other.fitness) return 1;
        if (this.fitness < other.fitness) return -1;
        return 0;
    }
    public void AddFitness(float fit)
    {
        this.fitness += fit;
    }
    public void SetFitness(float fit)
    {
        this.fitness = fit;
    }
    public float GetFitness()
    {
        return this.fitness;
    }
    

    //--------------COMPLEMENTARY METHODS---------------//
    protected void SetInputs(float[] inputs)
    {
        if (inputs.Length != neurons[0].Length)
            Debug.Log("The number of inputs received is " + inputs.Length + " and the number of input neurons is " + neurons[0].Length + ". Expect some inputs to be ignored!");

        for (int i = 0; i < neurons[0].Length; i++)
        {
            neurons[0][i] = inputs[i];
        }
    }
    static float SigmoidFunction(float value)
    {
        // Function is x = 1/(1 + e^(-x))
        return  (float)   1f / (1f + Mathf.Exp(-value));
    }
    static float NormalDistributionOf(float x, float sigma = 1f, float mu = 0f)
    {
        return (float)(1 / Math.Sqrt((2f * Math.PI * Mathf.Pow(sigma, 2f)))
                           * Mathf.Exp(-1f/2f * Mathf.Pow((x-mu)/sigma,2f)));
    }
    static float GetRandomNumberFromNormalDistribution()
    {
        return UnityEngine.Random.Range(-1f, 1f);
    }
  
    
    //--------------Setters & Getters------------------//
    public int[] GetLayers()
    {
        return layers;
    }
    public float[][][] GetWeights()
    {
        return weights;
    }
    public void SetLayers(int[] layers)
    {
        for (int i = 0; i < this.layers.Length; i++)
        {
            this.layers[i] = layers[i];
        }
    }
    public void SetWeights(float[][][] weights)
    {
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].Length; j++)
            {
                for (int k = 0; k < weights[i][j].Length; k++)
                {
                    this.weights[i][j][k] = weights[i][j][k];
                }
            }
        }
    }
}
public enum MutationStrategy
{
    best,
    Strategy1,
    Strategy2
}