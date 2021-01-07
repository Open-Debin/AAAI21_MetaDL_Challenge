""" This is a dummy baseline. It is just supposed to check if ingestion and 
scoring are called properly.
"""
import os
import logging
import csv
import gin
import torch
import numpy as np
from networks import Classifier_Triu as Classifier
from networks import Conv4 as feature_encoder
from helper import image_transform, LoadParameter, accuracy, mean_confidence_interval
import torch.optim as optim
import pdb
import tensorflow as tf
import torch.nn.functional as F

from metadl.api.api import MetaLearner, Learner, Predictor
@gin.configurable
class MyMetaLearner(MetaLearner):

    def __init__(self, meta_iterations, embedding_dim=64):
        super().__init__()
        self.embedding_fn = feature_encoder(num_classes=embedding_dim)
#         self.embedding_fn = LoadParameter(self.embedding_fn, torch.load('resnet18_tiered.pth.tar')['state_dict'])
        self.classifier = Classifier(fea_dim=embedding_dim).to(torch.device("cuda"))
#         self.classifier = Classifier(fea_dim=576).to(torch.device("cuda"))
        self.embedding_fn = self.embedding_fn.to(torch.device("cuda"))
        torch.cuda.empty_cache()
        self.optimizer =  optim.SGD(self.classifier.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=False)
        self.lr_schedule = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5500], gamma=0.1)
        self.count=0
        self.meta_iterations = meta_iterations
    def meta_fit(self, meta_dataset_generator) -> Learner:
        """
        Args:
            meta_dataset_generator : a DataGenerator object. We can access 
                the meta-train and meta-validation episodes via its attributes.
                Refer to the metadl/data/dataset.py for more details.
        
        Returns:
            MyLearner object : a Learner that stores the meta-learner's 
                learning object. (e.g. a neural network trained on meta-train
                episodes)
        """
        meta_train_dataset = meta_dataset_generator.meta_train_pipeline
        meta_valid_dataset = meta_dataset_generator.meta_valid_pipeline

        meta_train_dataset = meta_train_dataset.batch(1)
#         meta_valid_dataset = meta_valid_dataset.batch(2)
        logging.info('Starting meta-fit for the proto-net ...')
        self.embedding_fn.eval()
        acc_list = []
        
        for tasks_batch in meta_train_dataset :
            sup_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][1], tasks_batch[0][0]))
            que_set = tf.data.Dataset.from_tensor_slices(\
                (tasks_batch[0][4], tasks_batch[0][3]))

            new_ds = tf.data.Dataset.zip((sup_set, que_set))
            self.count+=1
            for ((spt_labs, spt_img), (qry_labs, qry_img)) in new_ds:

                spt_img = torch.tensor(np.array(spt_img)).permute(0,3,1,2).contiguous().to('cuda', non_blocking=True)
                spt_labs = torch.tensor(np.array(spt_labs)).to('cuda', non_blocking=True)
                with torch.no_grad():
                    spt_fea = self.embedding_fn(spt_img, True)[0]
                self.classifier.fit(spt_fea, spt_labs)

                qry_img = torch.tensor(np.array(qry_img)).permute(0,3,1,2).contiguous().to('cuda', non_blocking=True)
                qry_labs = torch.tensor(np.array(qry_labs)).to('cuda', non_blocking=True)
                torch.cuda.empty_cache()

                prob_list = []
                for a_img in qry_img:
                    a_img = a_img.unsqueeze(0)
                    with torch.no_grad():
                        a_img = self.embedding_fn(a_img, True)[0]
                    prob_list.append(self.classifier.predict(a_img))
                    probs = torch.cat(prob_list)
#             pdb.set_trace()
            acc_list.append(accuracy(probs.max(dim=1)[1], qry_labs))
            # Backward Propagation
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()
            qry_labs = torch.tensor(qry_labs, dtype=torch.long).cuda()
            ce_loss = F.cross_entropy(probs, qry_labs)
            torch.cuda.empty_cache()
            ce_loss.backward()
            torch.cuda.empty_cache()
            self.optimizer.step()
            torch.cuda.empty_cache()
            self.lr_schedule.step()
            if self.count%50 == 0:
                acc, h_ = mean_confidence_interval(acc_list)
                acc = round(acc * 100, 2)
                h_ = round(h_ * 100, 2)
                print("meta_test aver_accuracy:", acc, "h:", h_)
                acc_list = []
            if(self.count > self.meta_iterations):
                break

                
        return MyLearner(embedding_fn=self.embedding_fn, classifier=self.classifier)

@gin.configurable
class MyLearner(Learner):

    def __init__(self, embedding_dim=64, embedding_fn=None, classifier=None):
        super().__init__()
        if embedding_fn == None:
            self.embedding_fn = feature_encoder(num_classes=embedding_dim)
            self.embedding_fn = self.embedding_fn.cuda()
        else:
            self.embedding_fn = embedding_fn
            
        if classifier == None: 
            self.classifier  = Classifier(fea_dim=embedding_dim).to(torch.device("cuda"))
#             self.classifier = Classifier(fea_dim=576).to(torch.device("cuda"))
            self.classifier = self.classifier.cuda()
        else:
            self.classifier = classifier
        torch.cuda.empty_cache()
    def save(self, model_dir):
        """ Saves the learning object associated to the Learner. It could be 
        a neural network for example. 

        Note : It is mandatory to write a file in model_dir. Otherwise, your 
        code won't be available in the scoring process (and thus it won't be 
        a valid submission).
        """
#         print('save eva')
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        
#         Save a file for the code submission to work correctly.
        clf_file = os.path.join(model_dir, 'learner_clf.ckpt')
        embedding_file = os.path.join(model_dir, 'learner_encoder.ckpt')
        torch.save(self.classifier.state_dict(), clf_file)
        torch.save(self.embedding_fn.state_dict(), embedding_file)
            
    def load(self, model_dir):
#         print('load eva')
        """ Loads the learning object associated to the Learner. It should 
        match the way you saved this object in save().
        """
        if(os.path.isdir(model_dir) != True):
            raise ValueError(('The model directory provided is invalid. Please'
                + ' check that its path is valid.'))
        self.classifier.load_state_dict(torch.load(os.path.join(model_dir, 'learner_clf.ckpt')))
        self.embedding_fn.load_state_dict(torch.load(os.path.join(model_dir, 'learner_encoder.ckpt')))
        
        self.classifier = self.classifier.to(torch.device("cuda"))
        self.embedding_fn = self.embedding_fn.to(torch.device("cuda"))
        
    def fit(self, dataset_train) -> Predictor:
        """
        Args: 
            dataset_train : a tf.data.Dataset object. It is an iterator over 
                the support examples.
        Returns:
            ModelPredictor : a Predictor.
        """
#         print('fit eva')
        self.embedding_fn.eval()
        with torch.no_grad():
            for image, label in dataset_train:
                torch.cuda.empty_cache()
                image = self.tf2torch_tensor(image).to('cuda', non_blocking=True)
                label = torch.tensor(np.array(label)).to('cuda', non_blocking=True)
                spt_fea = self.embedding_fn(image, True)[0]
                self.classifier.fit(spt_fea, label)
            
        return MyPredictor(self.embedding_fn, self.classifier)
    
    def tf2torch_tensor(self, image):
        image = torch.tensor(np.array(image)).permute(0,3,1,2).contiguous()
#         image = F.interpolate(image, (42, 42), mode='bilinear', align_corners=True)
        return image
    
@gin.configurable   
class MyPredictor(Predictor):

    def __init__(self, embedding_fn, classifier):
        super().__init__()
#         print('predictor')
        self.embedding_fn = embedding_fn
        self.classifier = classifier
        torch.cuda.empty_cache()
    def predict(self, dataset_test):
        """ Predicts the label of the examples in the query set which is the 
        dataset_test in this case. The prototypes are already computed by
        the Learner.

        Args:
            dataset_test : a tf.data.Dataset object. An iterator over the 
                unlabelled query examples.
        Returns: 
            preds : tensors, shape (num_examples, N_ways). We are using the 
                Sparse Categorical Accuracy to evaluate the predictions. Valid 
                tensors can take 2 different forms described below.

        Case 1 : The i-th prediction row contains the i-th example logits.
        Case 2 : The i-th prediction row contains the i-th example 
                probabilities.

        Since in both cases the SparseCategoricalAccuracy behaves the same way,
        i.e. taking the argmax of the row inputs, both forms are valid.
        Note : In the challenge N_ways = 5 at meta-test time.
        """
        self.embedding_fn.eval()
        with torch.no_grad():
            for image in dataset_test:
                
                torch.cuda.empty_cache()
                image = self.tf2torch_tensor(image[0]).to('cuda', non_blocking=True)
#                 qry_fea = self.embedding_fn(image, True)[1]
#                 probs = tf.convert_to_tensor(self.classifier.predict(qry_fea).cpu().detach().numpy())
                prob_list = []
                for a_img in image:
                    a_img = a_img.unsqueeze(0)
                    a_img = self.embedding_fn(a_img, True)[0]
                    prob_list.append(self.classifier.predict(a_img).cpu().detach().numpy())
                    probs = np.concatenate(prob_list)
                probs = tf.convert_to_tensor(probs)
        return probs
    
    def tf2torch_tensor(self, image):
        image = torch.tensor(np.array(image)).permute(0,3,1,2).contiguous()
#         image = F.interpolate(image, (42, 42), mode='bilinear', align_corners=True)
        return image

