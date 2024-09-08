import pandas as pd #importing the libaries I need
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import accuracy_score
'''
#1a)

data = pd.read_csv("SpotifyFeatures.csv", sep=',') #uploading the data and separating it with "," 
                                                
songs = data.shape[0]   #the number of songs is equal to the number of rows in the dataframe

print("The number of songs is",songs)

features = data.shape[1]    #the number of features is equal to the number og collumns in the dataframe

print("The number of features is",features)

'''
#1b)

data = pd.read_csv("SpotifyFeatures.csv", sep=',',usecols=[0,11 ,12])   #removing the collumns i dont need, keeping genre, loudnes og liveness

data_pop = np.array(data.drop(data[data["genre"] != "Pop"].index))    #making a new dataframe containing only pop songs 
data_classical = np.array(data.drop(data[data["genre"] != "Classical"].index)) #making a new dataframe containing only classical songs    

#starting the labeling process
ones = np.ones(data_pop.shape[0]) #making a numpy array with equal amount of ones as pop songs
zeros = np.zeros(data_classical.shape[0]) #making a numpy array with equal amount of zeros as classical songs

labeled_data_pop = np.insert(data_pop,0,ones,axis=1) #merging the pop songs and the ones, witch means we now have labeled the pop data
labled_data_classical = np.insert(data_classical,0,zeros,axis=1) #merging the classical songes and the zeros, so this data to has labeles too

antall_pop = data_pop.shape[0]  #number of pop songs is the same as the number of rows in the new dataframe
print("The number of pop songs is", antall_pop)   

antall_classical = data_classical.shape[0]  #number of classical songs is the same as number of rows in the new dataframe
print("The number of classical songs is", antall_classical)


#1c)

matrix_1 = np.vstack((data_pop, data_classical)) #vertical stacking data_pop and data_classical to make matrix 1
matrix_1 =matrix_1[:,1:]        #removing the first collumn so we now have a matrix with only 2 collumns, liveness and loudness 
print(matrix_1)

matrix_2 = np.concatenate((ones, zeros), axis=0)   #merging the ones and the zeros so that we have a vector with labels
print(matrix_2)

#splitting the dataset in 4, 2 training matrices and 2 test matrices, the spit is train=80%, test=20%
matrix_1_train, matrix_1_test, matrix_2_train, matrix_2_test = sklearn.model_selection.train_test_split(matrix_1,matrix_2,test_size=0.20,train_size=0.80)

#1d)

plt.scatter(data_pop[:,1],data_pop[:,2], s = 5, label="pop")    #plotting pop songs as dots
plt.scatter(data_classical[:,1], data_classical[:,2], s = 5, label="classical") #plotting classical songs as a different color dots
plt.xlabel("livenes")   #x-axis
plt.ylabel("loudness") #y-axis
plt.legend()
plt.show()

#2a)

def sigmoid(y):
    return 1 / (1 + np.exp(-np.array((y), dtype=np.float64)))   #defining the sigmoid function

#defining my own logistic discrimination classifier
def LogisticRegression_SGD(matrix_1_train, matrix_2_train, learning_rate = 0.001, iterations = 100):
    w = np.zeros((matrix_1_train.shape[1],1))    #making a vector with zerows the same length as matrix1_train
    B = 0   #defining a varalble B starting as 0
    loss_list = []  #empty list that is going to contain the loss
    iteration_list = [] #empty list that is going to contain number of iterations 

    for i in range(iterations):
        iteration_list.append(i)
        for num in range(len(matrix_2_train)):  #stochastic gradient descent, therefore we need an exstra for loop, because the 
            samp = matrix_1_train[num]          #parameters are updated for every sample itertation
        
            y = np.dot(w.T,samp) + B    #finding y for 1 sample at a time instead of the whole dataset like gradient descent do

            A = sigmoid(float(y[0]))     #A is the predicted y-value, the probability for one as a value 

            #partial derivatives because we want to minimize the loss function 
            dw = np.dot(A-matrix_2_train[num], samp.T).reshape(-1, 1)
            db = np.sum(A-matrix_2_train[num])

            #updating the weights    
            w = w - learning_rate*dw
            B = B - learning_rate*db
            
        
        y = np.dot(w.T, matrix_1_train.T) + B

        A = sigmoid(y[0])   #A is the predicted y-value, the probability for one as a value 

        #cross entropy loss function
        loss = -np.sum(matrix_2_train*np.log(A) + (1-matrix_2_train)*np.log(1-A))
        loss_list.append(loss)

        if(i%10 == 0):
            print("loss etter", i, "iteration is: ", loss)
            

    return w, B, loss_list, iteration_list

w, b, loss_list, iteration_list = LogisticRegression_SGD(matrix_1_train, matrix_2_train)   

#plotting the training error as a funktion of epochs/iterations
x_verdier = np.array(loss_list)
y_verdier = np.array(iteration_list)

plt.plot(x_verdier,y_verdier )
plt.xlabel("iterations")
plt.ylabel("training error")
plt.show()

#defining a function that decides if the datapoint is a one or a zero based on the probability
def predict(x):
    z = np.dot(w.T, x.T) + b
    sig = sigmoid(z)
    y_pred = np.where(sig >=0.5,1,0)   
    return y_pred

#predicted values
y_hat = predict(matrix_1_test)
y_hat2 = predict(matrix_1_train)

#accuracy of training set
accuracy2 = accuracy_score(matrix_2_train,y_hat2.flatten())
print("the accuracy on the training set is", accuracy2)

#2b

#acuracy of test set
accuracy = accuracy_score(matrix_2_test, y_hat.flatten())
print("The accuracy on the test set is", accuracy)

#3a)
#making a confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(matrix_2_test, y_hat.flatten()))









