import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class RBM(object):

    def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):

        self.visibleDimensions = visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
        self.batchSize = batchSize
        
                
    def Train(self, X):

        ops.reset_default_graph()

        self.MakeGraph()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            
            trX = np.array(X)
            for i in range(0, trX.shape[0], self.batchSize):
                self.sess.run(self.update, feed_dict={self.X: trX[i:i+self.batchSize]})

            print("Trained epoch ", epoch)


    def GetRecommendations(self, inputUser):
                 
        hidden = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        visible = tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.weights)) + self.visibleBias)

        feed = self.sess.run(hidden, feed_dict={ self.X: inputUser} )
        rec = self.sess.run(visible, feed_dict={ hidden: feed} )
        return rec[0]       

    def MakeGraph(self):

        tf.set_random_seed(0)
        
        # variablat per cdo graf, peshe dhe bias
        self.X = tf.placeholder(tf.float32, [None, self.visibleDimensions], name="X")
        
        # Iicializohen peshat ne menyre te rastesishme
        maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        
        self.hiddenBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32, name="hiddenBias"))
        self.visibleBias = tf.Variable(tf.zeros([self.visibleDimensions], tf.float32, name="visibleBias"))
        
    # kalimi para
        # kampionojme shtresen e fshehur
        # marrim tensoret e probabiliteteve të fshehura
        hProb0 = tf.nn.sigmoid(tf.matmul(self.X, self.weights) + self.hiddenBias)
        # kampione nga cdo shperndarje
        hSample = tf.nn.relu(tf.sign(hProb0 - tf.random_uniform(tf.shape(hProb0))))
        # i bashkojme
        forward = tf.matmul(tf.transpose(self.X), hSample)
        
    # kalimi prapa
        # Rindërtohett shtresa e dukshme nga kampionin e shtresës së fshehur
        v = tf.matmul(hSample, tf.transpose(self.weights)) + self.visibleBias
        
        # vleresimet qe mungojne
        vMask = tf.sign(self.X) 
        vMask3D = tf.reshape(vMask, [tf.shape(v)[0], -1, self.ratingValues]) 
        vMask3D = tf.reduce_max(vMask3D, axis=[2], keepdims=True)  
        
        # vektoret e vleresimeve per cdo set individual me 10 vlerat binare te yjeve
        v = tf.reshape(v, [tf.shape(v)[0], -1, self.ratingValues])
        vProb = tf.nn.softmax(v * vMask3D) # funksioni i aktivizimit
        vProb = tf.reshape(vProb, [tf.shape(v)[0], -1]) 
        # përcaktohet kalimi prapa dhe perditesohen biaset e fshehura 
        hProb1 = tf.nn.sigmoid(tf.matmul(vProb, self.weights) + self.hiddenBias)
        backward = tf.matmul(tf.transpose(vProb), hProb1)
    
        # cfare do beje cdo epoch.
        # drejtoni kalimet përpara dhe prapa dhe perditeson peshat
        weightUpdate = self.weights.assign_add(self.learningRate * (forward - backward))
        # perditeson biasin e fshehur, minimizon divergjencen ne nyjet e fshehura
        hiddenBiasUpdate = self.hiddenBias.assign_add(self.learningRate * tf.reduce_mean(hProb0 - hProb1, 0))
        # perditeson biasin e dukshems, minimizon divergjencen ne rezultatet e dukshme
        visibleBiasUpdate = self.visibleBias.assign_add(self.learningRate * tf.reduce_mean(self.X - vProb, 0))

        self.update = [weightUpdate, hiddenBiasUpdate, visibleBiasUpdate]
        
    