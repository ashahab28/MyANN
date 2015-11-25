package myPerceptron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Vector;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/*
 * Developed based on Tom Mitchell's Book
 * Machine Learning, 1999
 */

/**
 *
 * @author ahmadshahab
 */
public class SinglePerceptron extends AbstractClassifier{

    /* Attributes */
    
    /**
     * Learning Rate
     */
    private double learningRate;
    
    /**
     * Class Treshold
     */
    private int classTreshold;
    
    /**
     * Learning Rule
     */
    private char learningRule;
        
    /**
     * Input Vector
     */
    private ArrayList<Double> weightVector;
    
    /**
     * Comma separated String for each weight
     * Written from the first attribute - last attribute
     * ex: 1, 0, 3, 2
     */
    private String weightInitialitation;
    
    /**
     * Max number of iteration to be set for training stop condition
     */
    private int maxIteration;
    
    /**
     * The stop condition for training by checking the mean squared error
     */
    private double MSE_ERROR;
    
    /**
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * @return the validationTreshold
     */
    public int getClassTreshold() {
        return classTreshold;
    }

    /**
     * @param classTreshold the validationTreshold to set
     */
    public void setClassTreshold(int classTreshold) {
        this.classTreshold = classTreshold;
    }
     
    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * @return the learningRule
     */
    public char getLearningRule() {
        return learningRule;
    }

    /**
     * @param learningRule the learningRule to set
     */
    public void setLearningRule(char learningRule) {
        this.learningRule = learningRule;
    }

    /**
     * @return the weightInitialitation
     */
    public String getWeightInitialitation() {
        return weightInitialitation;
    }

    /**
     * @param weightInitialitation the weightInitialitation to set
     */
    public void setWeightInitialitation(String weightInitialitation) {
        this.weightInitialitation = weightInitialitation;
    }
    
    /**
     * @return the weightVector
     */
    public ArrayList<Double> getWeightVector() {
        return weightVector;
    }

    /**
     * @param weightVector the weightVector to set
     */
    public void setWeightVector(ArrayList<Double> weightVector) {
        this.weightVector = weightVector;
    }
    
    /**
     * @return the maxIteration
     */
    public int getMaxIteration() {
        return maxIteration;
    }

    /**
     * @param maxIteration the maxIteration to set
     */
    public void setMaxIteration(int maxIteration) {
        this.maxIteration = maxIteration;
    }
    
    /**
     * @return the MSE_ERROR
     */
    public double getMSE_ERROR() {
        return MSE_ERROR;
    }

    /**
     * @param MSE_ERROR the MSE_ERROR to set
     */
    public void setMSE_ERROR(double MSE_ERROR) {
        this.MSE_ERROR = MSE_ERROR;
    }
    
    /**
     * Class Constructor
     * Filled with default parameter
     */
    public SinglePerceptron(){
//        learningRate = 0.3;
//        classTreshold = 1;
//        learningRule = 'P';
//        weightInitialitation = "";
//        maxIteration = 10000;
//        MSE_ERROR = 0.1;
        weightVector = new ArrayList<>();
    }
    
    /**
     * Build the Single Perceptron Classifier
     * @param i
     * @throws Exception 
     */
    @Override
    public void buildClassifier(Instances i) throws Exception {
        // detect capability from the instances data
        getCapabilities().testWithFail(i);
        
        // Remove missing class
        Instances data = new Instances(i);
        data.deleteWithMissingClass();
        
        // TO-DO
        // NOMINAL ATTRIBUTE HANDLING
        // NOMINAL CLASS HANDLING
        // MULTILABEL?
        
        // build classifier based on given parameter
        // choose the training rule 
        switch(getLearningRule()){
            case 'P':
                perceptronTraining(i);
                break;
            case 'B':
                batchTraining(i);
                break;
            case 'I':
                incrementalTraining(i);
                break;
        }
    }
    
    /**
     * Build a classifier using Perceptron training rule
     * @param instances
     */
    public void perceptronTraining(Instances instances){
        
        // Weight random initialitation
        if (getWeightVector().isEmpty()){
            // Ignore last attribute which is class
            for(int attrib_i = 0; attrib_i < instances.numAttributes() - 1; attrib_i++){
                getWeightVector().add(0.0);
            }
        }
        // Add biase value to the last index of weight
        getWeightVector().add(0.0);
        int biaseIndex = getWeightVector().size() - 1;

        // TO-DO add MSE & Iteration Number here
        for(int n  = 0; n < getMaxIteration(); n++){
            // For each epoch
            System.out.println("Epoch: " + n);
            System.out.println("--------------------");
            for(int i = 0; i < instances.numInstances(); i++){
                Instance instance = instances.instance(i);
                double targetClass = instance.classValue();
                double sum = netFunction(instance);
                double targetPredicted = signFunction(sum);
                System.out.println("Predicted Target: " + targetPredicted);
                System.out.println("Real Target: " + targetClass);

                // Update Weight
                if (targetClass != targetPredicted){
                    // for each weight 
                    for(int w_i = 0; w_i < biaseIndex; w_i ++){
                        double old_weight = getWeightVector().get(w_i);
                        double differenceTarget = targetClass - targetPredicted;
                        double new_weight = old_weight + (getLearningRate() * instance.value(w_i) * differenceTarget);

                        getWeightVector().set(w_i, new_weight);
                    }
                    // Don't forget the biase, again!
                    double old_weight = getWeightVector().get(biaseIndex);
                    double differenceTarget = targetClass - targetPredicted;
                    double new_weight = old_weight + (getLearningRate() * 1 * differenceTarget);

                    getWeightVector().set(biaseIndex, new_weight);

                    System.out.print("New Weight: ");
                    for(double weight: weightVector){
                        System.out.print(weight + " ");
                    }
                    System.out.println();
                }
            }
            
            // Check MSE value
            double mseError = mseError(instances);
            System.out.println("MSE Error: " + mseError);
            
            // If satisfied, stop the training
            if(mseError <= getMSE_ERROR()) break;
            System.out.println("--------------------");
        }
        // System.exit(1);
    }
    
    /**
     * Build a classifier using Batch training rule
     * @param i 
     */
    public void batchTraining(Instances i){
        
    }
    
    /**
     * Build a classifier using Incremental training rule
     * @param i 
     */
    public void incrementalTraining(Instances i){
        
    }

    /**
     * Net function to calculate weight * input
     * @param instance
     * @return 
     */
    public double netFunction(Instance instance){
        double sum = 0.0;
        int biaseIndex = getWeightVector().size() - 1;
        
        // Don't forget to count the biase
        for(int attrib_i = 0; attrib_i < instance.numAttributes() - 1; attrib_i++){
            // Sum of Input * Weight
            // System.out.print(instance.value(attrib_i) + " ");
            sum +=  instance.value(attrib_i) * getWeightVector().get(attrib_i);
        }
        // Biase! Input of biase = 1
        sum += 1 * getWeightVector().get(biaseIndex);
        
        return sum;
    }
    
    public double signFunction(double sumValue){
        if (sumValue >= 0){
            return 1.0;
        }
        else{
            return 0.0;
        }
    }
    
    public double mseError(Instances instances){
        double mseError = 0.0;
        
        ArrayList<Double> errorList = new ArrayList<>();
        for(int i = 0; i < instances.numInstances(); i++){
            Instance instance = instances.instance(i);
            double targetClass = instance.classValue();
            double sum = netFunction(instance);
            double targetPredicted = signFunction(sum);
//            System.out.println(targetClass + "-" + targetPredicted);
            errorList.add(targetClass - targetPredicted);
        }
        for(double error:errorList){
            mseError += Math.pow(error, 2);
        }
        return mseError / 2;
    }
    
    /**
    * Classifies a given test instance using the decision tree.
    *
    * @param instance the instance to be classified
    * @return the classification
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
      if (instance.hasMissingValue()) {
        throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                                     + "please.");
      }
      double sum = netFunction(instance);
      double targetPredicted = signFunction(sum);
      return targetPredicted;
    }
    
    @Override
    public Capabilities getCapabilities(){
        Capabilities perceptronCapability = super.getCapabilities();
        perceptronCapability.disableAll();
        
        // Attribute type capability
        perceptronCapability.enable(Capability.NOMINAL_ATTRIBUTES);
        perceptronCapability.enable(Capability.NUMERIC_ATTRIBUTES);
        
        // Class capability
        perceptronCapability.enable(Capability.NOMINAL_CLASS);
        perceptronCapability.enable(Capability.NUMERIC_CLASS);
        
        // Minimum number of instances allowed to be use
        perceptronCapability.setMinimumNumberInstances(0);
        
        return perceptronCapability;
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
          setLearningRate((new Double(learningString)));
        } else {
          setLearningRate(0.3);
        }
        String valTreshold = Utils.getOption('C', options);
        if (valTreshold.length() != 0) {
            setClassTreshold(Integer.parseInt(valTreshold));
        } else {
          setClassTreshold(1);
        }
        String trainingRule = Utils.getOption('T', options);
        if ("P".equals(trainingRule) || "B".equals(trainingRule) || "I".equals(trainingRule)){
            setLearningRule(trainingRule.charAt(0));
        } else{
            setLearningRule('P');
        }
        String weightInit = Utils.getOption('I', options);
        if (weightInit.length() != 0){
            String[] weightList = weightInit.replaceAll("\\s+", "").split(",");
            
            for (String weight : weightList) {
                getWeightVector().add(Double.valueOf(weight));
            }
        } else{
            setWeightInitialitation((""));
        }
        String mseError = Utils.getOption('E', options);
        if (mseError.length() != 0) {
            setMSE_ERROR(Double.valueOf(mseError));
        } else {
            setMSE_ERROR(0.1);
        }
        String numIteration = Utils.getOption('N', options);
        if (numIteration.length() != 0) {
            setMaxIteration(Integer.parseInt(numIteration));
        } else {
            setMaxIteration(501);
        }
        super.setOptions(options);
        Utils.checkForRemainingOptions(options);
  }
    
    @Override
    public String[] getOptions() {

      Vector<String> options = new Vector<>();

      options.add("-L");
      options.add("" + getLearningRate());
      options.add("-C");
      options.add("" + getClassTreshold());
      options.add("-T");
      options.add("" + getLearningRule());
      options.add("-I");
      options.add("" + getWeightInitialitation());
      options.add("-E");
      options.add("" + getMSE_ERROR());
      options.add("-N");
      options.add("" + getMaxIteration());

      Collections.addAll(options, super.getOptions());

      return options.toArray(new String[0]);
    }
}
