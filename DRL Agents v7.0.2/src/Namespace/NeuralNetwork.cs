using DRLAgents;
using System.Collections;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Text;
using UnityEngine;
using System;
using System.IO;
using Unity.VisualScripting;
using UnityEditor;

namespace DRLAgents
{
    public class NeuralNetwork
    {
        public static ActivationFunctionType activation = ActivationFunctionType.Tanh;
        public static ActivationFunctionType outputActivation = ActivationFunctionType.Tanh;
        public static MutationType mutation = MutationType.Classic;
        public static InitializationFunctionType initialization = InitializationFunctionType.GaussianDistribution;

        protected int[] layers;
        protected float[][] neurons;
        protected float[][][] weights;
        protected float[][] biases;
        protected float fitness;

        public NeuralNetwork(int[] layers)
        {
            InitializeLayers(layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            fitness = 0f;

        }
        public NeuralNetwork(NeuralNetwork networkCopy)
        {
            InitializeLayers(networkCopy.layers);
            InitializeNeuronsAndBiases(false);
            InitializeWeights(false);
            SetWeightsWith(networkCopy.weights);
            SetBiasesWith(networkCopy.biases);
            SetFitness(networkCopy.GetFitness());
        }
        public NeuralNetwork(string fileText)
        {
            string[] fileLines = Regex.Split(fileText, "\n|\r|\r\n");

            //Get Layers Data
            string[] layersLineStr = fileLines[0].Split("n,");//one more read than neccesary
            int noLayers = layersLineStr.Length - 1;
            int[] layersLineInt = new int[noLayers];
            Functions.ArrayConversion.ConvertStrArrToIntArr(layersLineStr, ref layersLineInt);
            InitializeLayers(layersLineInt);
            InitializeNeuronsAndBiases(true);
            InitializeWeights(true);

            //Get Weights Data
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < noLayers; i++)//First and last lineStr are ignored ( first lineStr is layerFormat, last lineStr is Fitness)
            {
                //One lineStr here are the weightss on a single neuronsNumber
                List<float[]> weightsOnLayer = new List<float[]>();

                string[] lineStr = fileLines[i].Split("w,");// one more read than neccesary
                float[] lineInt = new float[lineStr.Length - 1];
                Functions.ArrayConversion.ConvertStrArrToFloatArr(lineStr, ref lineInt);


                //This probArr must be devided depeding on the previous neuronsNumber number of neurons
                int numNeurOnPrevLayer = layersLineInt[i - 1];
                float[] weightsOnNeuron = new float[numNeurOnPrevLayer];
                int count = 0;

                for (int j = 0; j < lineInt.Length; j++)
                {

                    ///Problema cand ajunge la line32.lenght -1, ultima adaugare nu o baga in weightss on Layer
                    if (count < numNeurOnPrevLayer)
                    {
                        weightsOnNeuron[count] = lineInt[j];
                        count += 1;
                    }
                    if (count == numNeurOnPrevLayer)//This is CORRECT -> if count surpassed the number of neurOnPrevLayer
                    {
                        weightsOnLayer.Add(weightsOnNeuron);
                        weightsOnNeuron = new float[numNeurOnPrevLayer];
                        count = 0;
                    }
                }

                weightsList.Add(weightsOnLayer.ToArray());
            }
            this.SetWeightsWith(weightsList.ToArray());

            //Get Biases Data
            for (int i = noLayers, lay = 0; i < noLayers * 2; i++, lay++)
            {
                string[] lineStr = fileLines[i].Split("b,");//one more read than neccesary

                string str = lay.ToString() + ":";
                float[] lineInt = new float[lineStr.Length - 1];
                Functions.ArrayConversion.ConvertStrArrToFloatArr(lineStr, ref lineInt);
                this.biases[lay] = lineInt;
            }

            //Get Fitness Data
            string fitStr = fileLines[fileLines.Length - 1];
            fitStr = fitStr.Substring(0, fitStr.Length - 3);
            float fit = float.Parse(fitStr);
            this.SetFitness(fit);
        }
        public static void WriteBrain(in NeuralNetwork net, TextAsset networkFile, string name = null, string customFolderPath = null)
        {
            string path;
            if(customFolderPath != null)
            {
                path = customFolderPath + name;
            }
            else if(networkFile == null || networkFile == null) // file doesn t exist
            {
                path = "Assets/Neural_Networks/";
                if(!Directory.Exists(path))
                    Directory.CreateDirectory(path);
                path+=name;
            }
            else //file already exists
            {
                path = AssetDatabase.GetAssetPath(networkFile);  
            }

            
            StringBuilder data = new StringBuilder();
            

            //AddLayers
            data.Append(string.Join("n,", net.GetLayers()));
            data.Append("n,\n");

            //AddWeights
            StringBuilder weightsSB = new StringBuilder();
            float[][][] weights = net.GetWeights();
            foreach (float[][] layWeights in weights)
            {
                for (int i = 0; i < layWeights.Length; i++)
                {
                    weightsSB.Append(string.Join("w,", layWeights[i]));
                    weightsSB.Append("w,");
                }
                weightsSB.Append("\n");
            }
            data.Append(weightsSB);

            //AddBiases
            StringBuilder biasesSB = new StringBuilder();
            float[][] biases = net.GetBiases();
            for (int i = 0; i < biases.Length; i++)
            {
                biasesSB.Append(string.Join("b,", biases[i]));
                if (i == 0)
                    biasesSB.Append("b, @input layer biases are never used\n");
                else biasesSB.Append("b,\n");
            }
            data.Append(biasesSB);

            data.Append(net.GetFitness().ToString());
            data.Append("fit");
            File.WriteAllText(path, data.ToString());
            
            AssetDatabase.Refresh();
        }

        public float[] ForwardPropagation(float[] inputs)
        {
            SetInputs(inputs);

            for (int l = 1; l < layers.Length; l++)
            {
                for (int n = 0; n < neurons[l].Length; n++)
                {
                    float value = biases[l][n];
                    for (int p = 0; p < neurons[l - 1].Length; p++)
                    {
                        value += weights[l - 1][n][p] * neurons[l - 1][p];
                    }

                    neurons[l][n] = value;

                    if (l != layers.Length - 1)
                        neurons[l][n] = Activate(value, false);
                    else if (outputActivation != ActivationFunctionType.SoftMax)
                        neurons[l][n] = Activate(value, true);



                }
                if (l == layers.Length - 1 && outputActivation == ActivationFunctionType.SoftMax)
                    Functions.Activation.SoftMax(ref neurons[layers.Length - 1]);
            }

            return neurons[neurons.Length - 1]; //Return the last neurons layer with their values (OUTPUT)
        }

        //-----------------INITIALIZATION-----------------//
        protected void InitializeLayers(int[] layers)
        {
            this.layers = new int[layers.Length];
            SetLayersWith(layers);
        }
        protected void InitializeNeuronsAndBiases(bool emptyBiases)
        {
            List<float[]> neuronsList = new List<float[]>();
            List<float[]> biasesList = new List<float[]>();
            foreach (int neur in layers)
            {
                neuronsList.Add(new float[neur]);
                biasesList.Add(new float[neur]);
            }
            neurons = neuronsList.ToArray();

            if (!emptyBiases)
                for (int i = 1; i < biasesList.Count; i++)
                    Initialize(biasesList[i]);
            biases = biasesList.ToArray();
        }
        protected void InitializeWeights(bool empty)
        {
            List<float[][]> weightsList = new List<float[][]>();
            for (int i = 1; i < layers.Length; i++)
            {
                List<float[]> weightsOnLayerList = new List<float[]>();
                int neuronsInPreviousLayer = layers[i - 1];

                for (int n = 0; n < neurons[i].Length; n++)
                {
                    float[] neuronWeights = new float[neuronsInPreviousLayer];
                    if (!empty)
                        Initialize(neuronWeights);
                    weightsOnLayerList.Add(neuronWeights);
                }
                weightsList.Add(weightsOnLayerList.ToArray());
            }
            weights = weightsList.ToArray();
        }

        protected void Initialize(float[] axons)
        {
            //this is not a ref parameter because axons are inside a List so are referenced directly
            switch (initialization)
            {
                case InitializationFunctionType.GaussianDistribution:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.Initialization.RandomInNormalDistribution(0, 1);
                    }
                    break;
                case InitializationFunctionType.OtherDistribution1:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.Initialization.RandomValueInCustomDeviationDistribution(0.15915f, 1.061f, 0.3373f);
                    }
                    break;
                case InitializationFunctionType.OtherDistribution2:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.Initialization.RandomValueInCustomDeviationDistribution(0.15915f, 2f, 0.3373f);
                    }
                    break;
                case InitializationFunctionType.RandomValue:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.Initialization.RandomValue();
                    }
                    break;
                default:
                    for (int i = 0; i < axons.Length; i++)
                    {
                        axons[i] = Functions.Initialization.RandomValueInCustomDeviationDistribution(0.15915f, 2f, 0.3373f);
                    }
                    break;

            }
        }

        //--------------------MUTATIONS---------------------//
        public void MutateWeightsAndBiases()
        {
            for (int i = 0; i < weights.Length; i++)
                for (int j = 0; j < weights[i].Length; j++)
                    for (int k = 0; k < weights[i][j].Length; k++)
                        MutateOneWeightOrBias(ref weights[i][j][k]);

            for (int i = 1; i < biases.Length; i++)  //input biases are not mutated*
                for (int j = 0; j < biases[i].Length; j++)
                    MutateOneWeightOrBias(ref biases[i][j]);
        }
        protected void MutateOneWeightOrBias(ref float weightOrBias)
        {

            switch (mutation)
            {
                case MutationType.Classic:
                    Functions.Mutation.ClassicMutation(ref weightOrBias);
                    break;
                case MutationType.LightPercentage:
                    Functions.Mutation.LightPercentageMutation(ref weightOrBias);
                    break;
                case MutationType.LightValue:
                    Functions.Mutation.LightValueMutation(ref weightOrBias);
                    break;
                case MutationType.StrongPercentage:
                    Functions.Mutation.StrongPercentagegMutation(ref weightOrBias);
                    break;
                case MutationType.StrongValue:
                    Functions.Mutation.StrongValueMutation(ref weightOrBias);
                    break;
                case MutationType.Chaotic:
                    Functions.Mutation.ChaoticMutation(ref weightOrBias);
                    break;
                default:
                    Functions.Mutation.ClassicMutation(ref weightOrBias);
                    break;
            }
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

        //--------------------------ACTIVATION---------------------------//
        static float Activate(float value, bool isOutputLayer = false)
        {
            ///does not work for softMax because it is called separately
            ActivationFunctionType where = isOutputLayer == true ? outputActivation : activation;

            switch (where)
            {
                case ActivationFunctionType.Tanh:
                    return Functions.Activation.Tanh(value);
                case ActivationFunctionType.Sigmoid:
                    return Functions.Activation.Sigmoid(value);
                case ActivationFunctionType.Relu:
                    return Functions.Activation.ReLU(value);
                case ActivationFunctionType.LeakyRelu:
                    return Functions.Activation.LeakyReLU(value);
                case ActivationFunctionType.BinaryStep:
                    return Functions.Activation.BinaryStep(value);
                case ActivationFunctionType.Silu:
                    return Functions.Activation.SiLU(value);
                default:
                    return 0f;
            }
        }
        static float ActivationDerivative(float value, bool isOutputLayer)
        {
            ActivationFunctionType where = isOutputLayer == true ? outputActivation : activation;

            switch (where)
            {
                case ActivationFunctionType.Tanh:
                    return Functions.Derivatives.DerivativeTanh(value);
                case ActivationFunctionType.Sigmoid:
                    return Functions.Derivatives.DerivativeSigmoid(value);
                case ActivationFunctionType.Relu:
                    return Functions.Derivatives.DerivativeReLU(value);
                case ActivationFunctionType.LeakyRelu:
                    return Functions.Derivatives.DerivativeLeakyReLU(value);
                case ActivationFunctionType.BinaryStep:
                    return Functions.Derivatives.DerivativeBinaryStep(value);
                case ActivationFunctionType.Silu:
                    return Functions.Derivatives.DerivativeSiLU(value);
                default:
                    return 0;
            }


        }

        //-------------- GETTERS------------------//
        public int[] GetLayers()
        {
            return layers;
        }
        public float[][][] GetWeights()
        {
            return weights;
        }
        public float[][] GetBiases()
        {
            return biases;
        }
        public int GetInputsNumber()
        {
            return layers[0];
        }
        public int GetOutputsNumber()
        {
            return layers[layers.Length - 1];
        }

        public float GetWeight(int weightsLayer, int neuronIndex, int weightIndex)
        {
            return weights[weightsLayer][neuronIndex][weightIndex];
        }
        public float GetBias(int biasesLayer, int neuron)
        {
            return biases[biasesLayer][neuron];
        }
        public void SetWeight(int weightLayer, int neuronIndex, int weightIndex, float value)
        {
            weights[weightLayer][neuronIndex][weightIndex] = value;
        }
        public void SetBias(int biasLayer, int neuron, float value)
        {
            biases[biasLayer][neuron] = value;
        }
        //------------------SETTERS---------------//
        public void SetLayersWith(int[] layers)
        {
            for (int i = 0; i < this.layers.Length; i++)
            {
                this.layers[i] = layers[i];
            }
        }
        public void SetWeightsWith(float[][][] newWeights)
        {
            for (int i = 0; i < newWeights.Length; i++)
                for (int j = 0; j < newWeights[i].Length; j++)
                    for (int k = 0; k < newWeights[i][j].Length; k++)
                        this.weights[i][j][k] = newWeights[i][j][k];
        }
        public void SetBiasesWith(float[][] newBiases)
        {
            for (int i = 0; i < newBiases.Length; i++)
                for (int j = 0; j < newBiases[i].Length; j++)
                    this.biases[i][j] = newBiases[i][j];
        }
        //------------------------------------------------------------------------HEURISTIC ------------------------------------------------------------------------------//
        #region HEURISTIC

        float[][][] weightsGradients = null;//same form as weights
        float[][] biasesGradients = null;//same form as biases (input layer biases are not counted)

        float[][][] weightsVelocities = null;
        float[][] biasesVelocities = null;
        class BatchError
        {
            //batch error is stored inside a object referenced type just to be locked in multithreading
            public float error = 0;
            public BatchError()
            {
                error = 0;
            }
        }
        BatchError batchError = new BatchError();
        public void InitMainGradientsArrays()
        {
            if (weightsGradients != null)
                return;
            ///Init Gradients ARRAYS

            weightsGradients = new float[weights.Length][][];
            biasesGradients = new float[biases.Length][];

            for (int i = 0; i < weights.Length; i++)
            {
                weightsGradients[i] = new float[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++)

                    weightsGradients[i][j] = new float[weights[i][j].Length];
            }
            for (int i = 0; i < biases.Length; i++)
                biasesGradients[i] = new float[biases[i].Length];


            //Init Velocities ARRAYS
            if (weightsVelocities != null)
                return;

            weightsVelocities = new float[weights.Length][][];
            biasesVelocities = new float[biases.Length][];

            for (int i = 0; i < weights.Length; i++)
            {
                weightsVelocities[i] = new float[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++)
                    weightsVelocities[i][j] = new float[weights[i][j].Length];
            }
            for (int i = 0; i < biases.Length; i++)
                biasesVelocities[i] = new float[biases[i].Length];

        }
        public void ApplyGradients(float learnRate, int batchSize, float momentum, float regularization)
        {
            //apply the weightsGradients and biasesGradients with the specific learnRate
            for (int i = 0; i < weights.Length; i++)
                ApplyGradientsOnLayer(i, learnRate / batchSize);

            void ApplyGradientsOnLayer(int weightLayerIndex, float modifiedLearnRate)
            {
                int inNeurons = layers[weightLayerIndex];
                int outNeurons = layers[weightLayerIndex + 1];
                int biasLayerIndex = weightLayerIndex + 1;

                //weight decay used in regularization
                float weightDecay = 1 - regularization * modifiedLearnRate;

                for (int i = 0; i < outNeurons; i++)
                {
                    for (int j = 0; j < inNeurons; j++)
                    {
                        //update velocity
                        weightsVelocities[weightLayerIndex][i][j] = weightsVelocities[weightLayerIndex][i][j] * momentum - weightsGradients[weightLayerIndex][i][j] * modifiedLearnRate;
                        //update weight
                        weights[weightLayerIndex][i][j] = weights[weightLayerIndex][i][j] * weightDecay + weightsVelocities[weightLayerIndex][i][j];
                        //reset gradient
                        weightsGradients[weightLayerIndex][i][j] = 0;
                    }
                    //update velocity
                    biasesVelocities[biasLayerIndex][i] = biasesVelocities[biasLayerIndex][i] * momentum - biasesGradients[biasLayerIndex][i] * modifiedLearnRate;
                    //update bias
                    biases[biasLayerIndex][i] += biasesVelocities[biasLayerIndex][i];
                    //reset gradient
                    biasesGradients[biasLayerIndex][i] = 0;
                }
            }
        }


        public void UpdateGradients(float[] inputs, float[] desiredOutputs, LossFunctionType lossFunc)
        {
            float[][] localNeurons;
            Node[][] localNodes;

            //------------------------------------------------------------------

            //init arrays
            InitLocalNeurArray_and_SetInputs_and_InitLocalNodes();

            //calculate outputs using localNeurons
            float[] outs = CalculatePropagatedOutputs();

            //calculate output cost and update gradient for last weights
            float sampleErr = CalculateOutputNodesCost(outs, desiredOutputs);
            lock (batchError)
            {
                batchError.error += sampleErr;
            }
            UpdateGradientsForLayer(layers.Length - 2);

            //update gradients foreach layer
            //Update gradients for hidden weights
            for (int weightLayer = layers.Length - 3; weightLayer >= 0; weightLayer--)//parse each weights layer from END to BEGINING
            {
                CalculateNodesCost(weightLayer + 1);//Calculate the valuesInDerivative of the nodes from the left
                UpdateGradientsForLayer(weightLayer);//Get the modifications needed for each weight and bias in that layer
            }

            //------------------------------------------------------------------
            //METHODS USED
            void InitLocalNeurArray_and_SetInputs_and_InitLocalNodes()
            {
                //init array - do not modify here
                localNeurons = new float[neurons.Length][];
                for (int i = 0; i < localNeurons.Length; i++)
                    localNeurons[i] = new float[neurons[i].Length];

                //init inputs inside
                if (inputs.Length != localNeurons[0].Length)
                    Debug.LogError("<color=red>The number of inputs received is " + inputs.Length + " and the number of input neurons is " + localNeurons[0].Length + "</color>.<color=grey> Expect some inputs to be ignored!</color>");
                for (int i = 0; i < localNeurons[0].Length; i++)
                {
                    localNeurons[0][i] = inputs[i];
                }

                //init local nodes
                localNodes = new Node[layers.Length][];
                localNodes[0] = new Node[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                    localNodes[0][i].valueOut = inputs[i];

            }

            float[] CalculatePropagatedOutputs()
            {
                for (int l = 1; l < layers.Length; l++)
                {
                    //create nodes from neurons
                    localNodes[l] = new Node[localNeurons[l].Length];

                    for (int n = 0; n < localNeurons[l].Length; n++)
                    {

                        float value = biases[l][n];
                        int previousLayerNeuronsNumber = layers[l - 1];
                        for (int p = 0; p < previousLayerNeuronsNumber; p++)
                        {
                            value += weights[l - 1][n][p] * localNeurons[l - 1][p];
                        }

                        localNodes[l][n].valueIn = value;
                        localNeurons[l][n] = value;//is activated after

                        if (l < layers.Length - 1)
                            localNeurons[l][n] = Activate(value, false);
                        else if (outputActivation != ActivationFunctionType.SoftMax)
                            localNeurons[l][n] = Activate(value, true);

                        localNodes[l][n].valueOut = localNeurons[l][n];

                    }

                    ///SPECIAL CASE i apply soft max for neurons values (i change their result into statistics)
                    if (l == layers.Length - 1 && outputActivation == ActivationFunctionType.SoftMax)
                    {
                        int neuronsOnLastLayer = layers[l];

                        //Get values In  (it works also for values out because the values are passed normally without Activation =: softMax activation is made after all node values are known)
                        float[] valuesIn = new float[neuronsOnLastLayer];

                        for (int n = 0; n < localNeurons[l].Length; n++)
                            valuesIn[n] = localNodes[l][n].valueIn;

                        //Activate them
                        Functions.Activation.SoftMax(ref valuesIn);

                        //Set values Out
                        for (int n = 0; n < localNodes[l].Length; n++)
                        {
                            localNeurons[l][n] = valuesIn[n];
                            localNodes[l][n].valueOut = localNeurons[l][n];
                        }


                    }

                }
                return localNeurons[localNeurons.Length - 1];
            }
            float CalculateOutputNodesCost(float[] outputs, float[] expectedOutputs)
            {
                //calculates average error of output nodes
                //calculates output nodes costValue

                if (outputActivation == ActivationFunctionType.SoftMax)
                    return CalculateOutputNodesCostForSoftMax(outputs, expectedOutputs);

                float cost = 0;
                for (int i = 0; i < outputs.Length; i++)
                {

                    if (lossFunc == LossFunctionType.Quadratic)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.QuadraticDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(localNodes[localNodes.Length - 1][i].valueIn, true);
                        cost += Functions.Cost.Quadratic(outputs[i], expectedOutputs[i]);
                    }
                    else if (lossFunc == LossFunctionType.Absolute)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.AbsoluteDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(localNodes[localNodes.Length - 1][i].valueIn, true);
                        cost += Functions.Cost.Absolute(outputs[i], expectedOutputs[i]);
                    }
                    else if (lossFunc == LossFunctionType.CrossEntropy)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.CrossEntropyDerivative(outputs[i], expectedOutputs[i]) * ActivationDerivative(localNodes[localNodes.Length - 1][i].valueIn, true);
                        float localCost = Functions.Cost.CrossEntropy(outputs[i], expectedOutputs[i]);
                        cost += float.IsNaN(localCost) ? 0 : localCost;
                    }

                }

                return cost / outputs.Length;//divided by the number of neurons
            }//LOSS FUNCTION
            float CalculateOutputNodesCostForSoftMax(float[] outputs, float[] expectedOutputs)
            {
                float cost = 0f;
                float[] derivatedInValues = new float[outputs.Length];//the InValues that must be used afterwards to be activated
                for (int i = 0; i < derivatedInValues.Length; i++)
                    derivatedInValues[i] = localNodes[localNodes.Length - 1][i].valueIn;
                Functions.Derivatives.DerivativeSoftMax(ref derivatedInValues);

                for (int i = 0; i < outputs.Length; i++)
                {
                    if (lossFunc == LossFunctionType.Quadratic)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.QuadraticDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                        cost += Functions.Cost.Quadratic(outputs[i], expectedOutputs[i]);
                    }
                    else if (lossFunc == LossFunctionType.CrossEntropy)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.CrossEntropyDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                        float localCost = Functions.Cost.CrossEntropy(outputs[i], expectedOutputs[i]);
                        cost += float.IsNaN(localCost) ? 0 : localCost;
                    }
                    else if (lossFunc == LossFunctionType.Absolute)
                    {
                        localNodes[localNodes.Length - 1][i].costValue = Functions.Cost.AbsoluteDerivative(outputs[i], expectedOutputs[i]) * derivatedInValues[i];
                        cost += Functions.Cost.Absolute(outputs[i], expectedOutputs[i]);
                    }
                }

                return cost;
            }//LOSS FUNCTION USED WHEN SOFTMAX USED
            void CalculateNodesCost(int neuronsLayer)
            {
                //IT DOES NOT APPLY FOR OUTPUT NEURON LAYER and INPUT LAYER
                if (neuronsLayer == 0)
                    return;
                int nodesNum = localNodes[neuronsLayer].Length;
                int nextLayerNeuronsNum = layers[neuronsLayer + 1];

                //The node value is equal to:
                // = Sum(nextLayerNeuron * connectionWeight) * Activation'(nodeValue.valueIN);

                for (int i = 0; i < nodesNum; i++)
                {
                    localNodes[neuronsLayer][i].costValue = 0;
                    for (int j = 0; j < nextLayerNeuronsNum; j++)
                    {
                        localNodes[neuronsLayer][i].costValue += localNodes[neuronsLayer + 1][j].costValue * weights[neuronsLayer][j][i]; // sum of each nextNeuron*connectionWeight;
                    }

                    localNodes[neuronsLayer][i].costValue *= ActivationDerivative(localNodes[neuronsLayer][i].valueIn, false);
                }
            }
            void UpdateGradientsForLayer(int weightLayerIndex)
            {
                //this is a weight layer
                //node layer is always + 1 up
                int inNeurons = layers[weightLayerIndex];
                int outNeurons = layers[weightLayerIndex + 1];
                int biasLayerIndex = weightLayerIndex + 1;
                lock (weightsGradients)
                {
                    for (int i = 0; i < outNeurons; i++)
                        for (int j = 0; j < inNeurons; j++)
                            weightsGradients[weightLayerIndex][i][j] += localNodes[weightLayerIndex][j].valueOut * localNodes[weightLayerIndex + 1][i].costValue;
                }
                lock (biasesGradients)
                {
                    for (int i = 0; i < outNeurons; i++)
                        biasesGradients[biasLayerIndex][i] += 1 * localNodes[weightLayerIndex + 1][i].costValue;
                }
                return;
            }
        }

        public float GetError()
        {
            //also resets the error
            float whatToReturn = batchError.error;
            batchError.error = 0;
            return whatToReturn;
        }
        //------------------------------------------------------------------------END HEURISTIC ------------------------------------------------------------------------------//
        #endregion




        //--------------COMPLEMENTARY METHODS---------------//
        protected void SetInputs(float[] inputs)
        {
            if (inputs.Length != neurons[0].Length)
                Debug.LogError("<color=red>The number of inputs received is " + inputs.Length + " and the number of input neurons is " + neurons[0].Length + "</color>.<color=grey> Expect some inputs to be ignored!</color>");


            for (int i = 0; i < neurons[0].Length; i++)
            {
                neurons[0][i] = inputs[i];
            }
        }


    }
}