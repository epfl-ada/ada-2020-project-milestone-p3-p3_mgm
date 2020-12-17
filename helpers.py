import numpy as np
import random
import torch
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef,f1_score

def last_act(entry):
    '''
        entry    -> the entry
        -------
        Returns 
        the ID of the season that is the last act of friendship
    '''
    return [ season['season'] for season in entry['seasons'][:-1] if 'support' in season['interaction'].values()][-1]
    
   
def seasons_before_betrayal(entry):
    '''
        entry    -> the entry
        -------
        Returns 
        list of seasons before the last act of friendship
    '''
    return [ season for season in entry['seasons'] if season['season'] <= last_act(entry)]



def get_messages(entry,from_who, i=None, j=None, lasting_4=True):
    '''
        entry      -> the entry
        from_who   -> from who of the players we want the messages, 'victim' or 'betrayer'
        i,j        -> for calculating average score for seasons[i:j]
        lasting_4  -> True : it will not include the entries that have less than 4 seasons
                      False: it will include all entries 
        -------
        Returns 
        list of lists of features for each message from from_who for the seasons(before betrayal)[i:j] for the given entry 
    '''
    seasons = seasons_before_betrayal(entry)
    if len(seasons)<4 and lasting_4:
        seasons=[]
    return [season['messages'][from_who] for season in seasons[i:j] 
            if (len(season['messages']['victim'])>0 and len(season['messages']['betrayer'])>0)]



def get_feature(entry,from_who,feature,i=None, j=None,additional=None,lexicon=False, lasting_4=True):
    '''
        eentry      -> the entry
        from_who   -> from who of the players we want the messages, 'victim' or 'betrayer'
        feature    -> the feature we want to extract
        i,j        -> for calculating average score for seasons[i:j]
        additional -> additional attribut for some features(ex. sentiment:positive, negative, neutral)
        lexicon    -> for extracting lexicon words features
        lasting_4  -> True : it will not include the entries that have less than 4 seasons
                      False: it will include all entries 
        -------
        Returns 
        list of average scores of feature for each message from seasons[i:j]
    '''
    if(additional is None and not lexicon):
        return [[message[feature] for message in s] for s in get_messages(entry,from_who,i,j,lasting_4=lasting_4)]
    if not lexicon:
        return [[message[feature][additional] for message in s] for s in get_messages(entry,from_who,i,j,lasting_4=lasting_4)]

    return [[len(message[feature][additional])for message in s if additional in message[feature].keys()] for s in get_messages(entry,from_who,i,j,lasting_4=lasting_4)]





def avg_feature_entries_perSeason(entry_set,from_who,feature,i=None, j=None,additional=None,lexicon=False,lasting_4=True):
    '''
        entry      -> the entry
        from_who   -> from who of the players we want the messages, 'victim' or 'betrayer'
        feature    -> the feature we want to extract
        i,j        -> for calculating average score for seasons[i:j]
        additional -> additional attribut for some features(ex. sentiment:positive, negative, neutral)
        lexicon    -> for extracting lexicon words features
        lasting_4  -> True : it will not include the entries that have less than 4 seasons
                      False: it will include all entries 
        -------
        Returns 
        list of average scores of feature for each message from seasons[i:j] for each entry
    '''
    avg_values = []
    for entry in entry_set:
        sentences_per_season = get_feature(entry,from_who,"n_sentences",i,j,additional=None,lexicon=False,lasting_4=lasting_4)
        if len(get_feature(entry,from_who,feature,i,j,additional,lexicon,lasting_4=lasting_4))>0:
            for t,s in enumerate(get_feature(entry,from_who,feature,i,j,additional,lexicon,lasting_4=lasting_4)):
                if lexicon:
                    avg_values.append(np.sum(s)/np.sum(sentences_per_season[t]))
                else:
                    avg_values.append(np.mean(s))
    return avg_values


def extract_features(entry,who,i,j,lasting_4):
    ''' 
        Extracts all the features from the given entries
        entry     -> the entries
        who       -> from who of the players we want the messages, 'victim' or 'betrayer'
        i,j       -> for calculating average score for seasons[i:j]
        lasting_4 -> True : it will not include the entries that have less than 4 seasons
                     False: it will include all entries 
        -------
        Returns
        item      -> list of all the average scores of the features 
    '''
    politness_item = avg_feature_entries_perSeason(entry,who,"politeness",i=i,j=j,lasting_4=lasting_4)
    requests_item = avg_feature_entries_perSeason(entry,who,"n_requests",i=i,j=j,lasting_4=lasting_4)
    words_item = avg_feature_entries_perSeason(entry,who,"n_words",i=i,j=j,lasting_4=lasting_4)
    sentences_item = avg_feature_entries_perSeason(entry,who,"n_sentences",i=i,j=j,lasting_4=lasting_4)
    positive_item = avg_feature_entries_perSeason(entry,who,"sentiment",i=i,j=j,additional="positive",lasting_4=lasting_4)
    negative_item = avg_feature_entries_perSeason(entry,who,"sentiment",i=i,j=j,additional="negative",lasting_4=lasting_4)
    neutral_item = avg_feature_entries_perSeason(entry,who,"sentiment",i=i,j=j,additional="neutral",lasting_4=lasting_4)
    premise_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="premise",lexicon=True,lasting_4=lasting_4)
    claim_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="claim",lexicon=True,lasting_4=lasting_4)
    comparison_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="disc_comparison",lexicon=True,lasting_4=lasting_4)
    expansion_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="disc_expansion",lexicon=True,lasting_4=lasting_4)
    contingency_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="disc_contingency",lexicon=True,lasting_4=lasting_4)
    temporal_future_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="disc_temporal_future",lexicon=True,lasting_4=lasting_4)
    temporal_rest_item = avg_feature_entries_perSeason(entry,who,"lexicon_words",i=i,j=j,additional="disc_temporal_rest",lexicon=True,lasting_4=lasting_4)

    item = [politness_item,positive_item,negative_item,neutral_item,requests_item,words_item,sentences_item,\
                premise_item,claim_item,comparison_item,expansion_item,contingency_item,temporal_future_item,temporal_rest_item]
    return item


def standardize(x,mean=None,std=None,test=False):
    '''
        test -> False : used for standardizing the train data
            True  : used for standardizing test data
        Standardize the data with zero mean and unit variance
    '''
    if not test:
        mean=np.mean(x,axis=0)
        std = np.std(x,axis=0)
    return (x-mean)/std, mean, std

def batch(labels, features, batch_size):
    for i in range(0,len(labels),batch_size):
        np_labels = np.array(labels[i:i+batch_size])
        np_features = np.array(features[i:i+batch_size])
        indx = np.arange(0,np_features.shape[0])
        random.shuffle(indx)
        yield np_labels[indx], np_features[indx]
        
       
def train_model(model, labels, features,X_validation, y_validation, no_epoch, batch_size, optimizer, criterion,cuda_is_available, verbose = True):
    accuracy_v = []
    accuracy_t = []
    f1_scores = []
    matthews_corrcoefs = []
    for epoch in range(no_epoch):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        i = 0
        for lb, ft in batch(labels, features,batch_size):
            #get the inputs; data is a list of [inputs, labels]
            sample = torch.from_numpy(ft).float().cuda()
            target = torch.from_numpy(lb).long().cuda()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            out = model(sample)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            # print statistics
            if verbose:
                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %f' %
                          (epoch + 1, i, running_loss/100))
                    running_loss = 0.0
                i += 1
                
        #validation
        #Perform the same operations as above but without calculating the gradient
        model.eval()
        with torch.no_grad():
            predictions_v = None
            predictions_t = None
            if cuda_is_available:
                predictions_v = predict_label(model, torch.from_numpy(X_validation).float().cuda())
                predictions_t = predict_label(model, torch.from_numpy(features).float().cuda())
            else:
                predictions_v = predict_label(model, torch.from_numpy(X_validation).float())
                predictions_t = predict_label(model, torch.from_numpy(features).float())

            accuracy_v.append(np.mean(predictions_v == y_validation))
            accuracy_t.append(np.mean(predictions_t == labels))
            matthews_corrcoefs.append(matthews_corrcoef(y_validation, predictions_v))
            f1_scores.append(f1_score(y_validation, predictions_v, average='weighted'))
            
    print('Finished Training')
    return accuracy_v,accuracy_t,f1_scores,matthews_corrcoefs
 
#Compares the two outcomes from the network and assigns a label according to the bigger value    
def predict_label(model, features):
    with torch.no_grad():
        model.eval()
        output = model(features)
        output = output.detach().cpu().numpy()
        return np.array([1 if label[1]>=label[0] else 0 for label in output ])
    
    
def cross_validation(net_first_layer_size, labels, features, k_fold, no_epoch, batch_size,lr,cuda_is_available, apply_smote=False,save_model=None):
    kf = KFold(n_splits=k_fold)
    accuracy_v = []
    accuracy_t = []
    f1_scores = []
    matthews_corrcoefs = []
    
    accuracy_v_epoch = []
    accuracy_t_epoch = []
    f1_scores_epoch = []
    matthews_corrcoefs_epoch = []
    
    ms_arr = []
    i = 0
    for train_index, validation_index in kf.split(features):
        #Split the dataset into train and validation sets
        X_train, X_validation = features[train_index], features[validation_index]
        y_train, y_validation = labels[train_index], labels[validation_index]
        #Correct the unbalanced nature of our dataset. Not always used.
        if apply_smote:
            oversample = SMOTE()
            X_train, y_train = oversample.fit_sample(X_train, y_train)
            rand_idx=np.random.permutation(len(X_train))
            X_train = X_train[rand_idx]
            y_train = y_train[rand_idx]
        #Apply standardization to both sets
        X_train,mean,std = standardize(X_train)
        X_validation,_,_ = standardize(X_validation,mean,std,True)
        #Use cuda in order to speed up the computation if tis available
        if cuda_is_available:
            model = Net(net_first_layer_size).cuda()
        else:
            model = Net(net_first_layer_size)
        #Set the Loss functions and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr)

        #Calculate the evaluation metrics such as accuracy and f1 score
        accuracy_ve,accuracy_te,f1_scorese,matthews_corrcoefse = train_model(model, y_train, X_train, X_validation, y_validation, no_epoch, batch_size, optimizer, criterion,cuda_is_available,True)
        accuracy_v_epoch.append(accuracy_ve)
        accuracy_t_epoch.append(accuracy_te)
        f1_scores_epoch.append(f1_scorese)
        matthews_corrcoefs_epoch.append(matthews_corrcoefse)
        
        predictions_v = None
        predictions_t = None
        if cuda_is_available:
            predictions_v = predict_label(model, torch.from_numpy(X_validation).float().cuda())
            predictions_t = predict_label(model, torch.from_numpy(X_train).float().cuda())
        else:
            predictions_v = predict_label(model, torch.from_numpy(X_validation).float())
            predictions_t = predict_label(model, torch.from_numpy(X_train).float())
        
        accuracy_v.append(np.mean(predictions_v == y_validation))
        accuracy_t.append(np.mean(predictions_t == y_train))
        matthews_corrcoefs.append(matthews_corrcoef(y_validation, predictions_v))
        f1_scores.append(f1_score(y_validation, predictions_v, average='weighted'))
        ms_arr.append([mean, std])
        if save_model is not None:
            torch.save(model, f'model_{save_model}_{i}')
            i = i+1
    return (accuracy_v, accuracy_t, f1_scores, matthews_corrcoefs),(accuracy_v_epoch,accuracy_t_epoch,f1_scores_epoch,matthews_corrcoefs_epoch), ms_arr

##Our Fully connected Linear Neural Network Model
class Net(nn.Module):
    def __init__(self,first_layer_size):
        super(Net, self).__init__()
        self.first_layer_size = first_layer_size
        
        
        self.model = nn.Sequential(nn.Linear(first_layer_size, 64),
                                   nn.ReLU(0.1),
                                   nn.Linear(64, 32),
                                   nn.ReLU(0.1),
                                   nn.Linear(32, 2))
        
    def forward(self,x):
        return self.model(x)
