from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Models class that takes test file as a parameter
class ModelSelector:
    
    def __init__(self,data_file=None):
        try:
            self.test_data=pd.read_csv(data_file) #data_file
        except FileNotFoundError:
            print("File not found!")
            return None
        
        #setup the train dataset to train the models when called.
        self.train_data=pd.read_pickle('selected_features.pkl')
        self.train_data_X=self.train_data.drop('y',axis=1)
        self.train_data_Y=self.train_data[['y']]
        
        
    
    #data encoder function
    #encodes the test dataset into appropriate encoding so as to match the format of train dataset.
    #returns an encoded dataset
    def data_encoder(self):
        l_encoder_columns=['housing','loan','y']
        one_hot_encoder_columns=['job','marital','contact','poutcome']
        int_columns=['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
        data_encoded=self.test_data.loc[:,int_columns+l_encoder_columns]
        
        #Label encoding
        le=LabelEncoder()
        for col in l_encoder_columns:
            data_encoded[col]=le.fit_transform(data_encoded[col])
        
        #one-hot encoding
        onehot_en=OneHotEncoder(dtype=int,drop='first')
        one_hot_encoded=onehot_en.fit_transform(self.test_data[one_hot_encoder_columns])
        one_hot_encoded=one_hot_encoded.toarray()
        one_hot_encoded=pd.DataFrame(one_hot_encoded,columns=onehot_en.get_feature_names_out(one_hot_encoder_columns))
        
        #final encoded data
        data_encoded=pd.concat([data_encoded,one_hot_encoded],axis=1)
        return data_encoded[self.train_data.columns.to_list()]
    
    #this function uses MinMaxScaler to scale the dataset given as a parameter.
    #returns a scaled dataset.
    #
    def Scaler(self,data_frame):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data_frame)
    
    #cross-validation function that takes a model, features and target datasets as parameters
    #returns various cross-validation performance measures in a dictionary format.
    def cross_validation(self,model, X_input, Y_input, cv_fold=5):
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
        results = cross_validate(estimator=model,X=X_input,y=Y_input,cv=cv_fold,scoring=scoring_metrics,return_train_score=True)
        return {
                "Mean Training Accuracy": results['train_accuracy'].mean()*100,
                "Mean Training Precision": results['train_precision'].mean(),
                "Mean Training Recall": results['train_recall'].mean(),
                "Mean Training F1 Score": results['train_f1'].mean(),
                "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
                "Mean Validation Precision": results['test_precision'].mean(),
                "Mean Validation Recall": results['test_recall'].mean(),
                "Mean Validation F1 Score": results['test_f1'].mean()  
        }
    
    #builds a confusion matrix by taking actual target class, predicted target class and different classes in target feature as parameters.
    #plots the confusion matrix using the mlxtend package.
    def confusion_matrix_builder(self,Y,pred,class_l):
        cm=confusion_matrix(Y,pred,labels=class_l)
 
        fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap='Blues',colorbar=True)
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.show()
    
    #definition of logistic regression model, Random forest classifier, NaiveBayes classifier and k-nearest neighbour classifier.
    #every model takes 3 optional parameters:
    #cross_val: takes [False,True] values, by default set to false. when set true, returns cross-validation measures.
    #pred: takes [False, True] values, by default set to True. returns the confusion matrix using the test dataset.
    #each model returns a cross-validation performance metrices or confusion matrix or both based on the parameters set.
    
    def LogisticRegression_model(self,cross_val=False,pred=True):
        lr=LogisticRegression()
        scaled_x=self.Scaler(self.train_data_X)
        lr.fit(scaled_x,self.train_data_Y)
        if cross_val:
            lr_result= self.cross_validation(lr,scaled_x,self.train_data_Y,5)
            print(lr_result)
        if pred:
            test_data_encoded=self.data_encoder()
            test_data_encoded_X=test_data_encoded.drop('y',axis=1)
            test_data_encoded_Y=test_data_encoded[['y']]
            test_data_encoded_X_scaled=self.Scaler(test_data_encoded_X)
            self.confusion_matrix_builder(test_data_encoded_Y,lr.predict(test_data_encoded_X_scaled),lr.classes_)
            
        
    def RandomForest_model(self,cross_val=False,pred=True):
        scaled_x=self.Scaler(self.train_data_X)
        rf=RandomForestClassifier(min_samples_split=0.05,min_samples_leaf=5,criterion='entropy',random_state=0,max_depth=10)
        rf.fit(scaled_x,self.train_data_Y)
        
        if cross_val:
            rf_result= self.cross_validation(rf,scaled_x,self.train_data_Y,5)
            print(rf_result)
        if pred:
            test_data_encoded=self.data_encoder()
            test_data_encoded_X=test_data_encoded.drop('y',axis=1)
            test_data_encoded_Y=test_data_encoded[['y']]
            test_data_encoded_X_scaled=self.Scaler(test_data_encoded_X)
            self.confusion_matrix_builder(test_data_encoded_Y,rf.predict(test_data_encoded_X_scaled),rf.classes_)
             
    def NaiveBayes_model(self,cross_val=False,pred=True):
        nb=GaussianNB()
        scaled_x=self.Scaler(self.train_data_X)
        nb.fit(scaled_x,self.train_data_Y)
        if cross_val:
            rf_result= self.cross_validation(nb,scaled_x,self.train_data_Y,5)
            print(rf_result)
        if pred:
            test_data_encoded=self.data_encoder()
            test_data_encoded_X=test_data_encoded.drop('y',axis=1)
            test_data_encoded_Y=test_data_encoded[['y']]
            test_data_encoded_X_scaled=self.Scaler(test_data_encoded_X)
            self.confusion_matrix_builder(test_data_encoded_Y,nb.predict(test_data_encoded_X_scaled),nb.classes_)
    
    def kNN_model(self,cross_val=False,pred=True):
        knn=KNeighborsClassifier(n_neighbors=2)
        scaled_x=self.Scaler(self.train_data_X)
        knn.fit(scaled_x,self.train_data_Y)
        if cross_val:
            rf_result= self.cross_validation(knn,scaled_x,self.train_data_Y,5)
            print(rf_result)
        if pred:
            test_data_encoded=self.data_encoder()
            test_data_encoded_X=test_data_encoded.drop('y',axis=1)
            test_data_encoded_Y=test_data_encoded[['y']]
            test_data_encoded_X_scaled=self.Scaler(test_data_encoded_X)
            self.confusion_matrix_builder(test_data_encoded_Y,knn.predict(test_data_encoded_X_scaled),knn.classes_)