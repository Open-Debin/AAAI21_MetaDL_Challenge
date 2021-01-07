import os
import pdb
import time
import pickle
import random
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from basic_code import util
from basic_code.configuration import args
from basic_code.classifiers_define import MetaQDA
from basic_code import task_test_generator as dataloder
from urt_code.lib.utils import convert_secs2time, AverageMeter, time_string
from urt_code.lib.config_utils import Logger

start = time.time()
SEED = 4603
print('random seed: '.format(SEED))
print("Sklearn verion is {}".format(sklearn.__version__))
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.deterministic = True
# CUDA_VISIBLE_DEVICES=6 python train_mqda.py
# def main(grid_search=False, *params_dict):
def main():
    args.device='1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    domain_net='cifarfs' # cub
    net='s2m2r'
    domain_data='cifarfs'
    args.lr = 4e-5
    args.support_shot=5
    args.feature_dim=640
    args.bool_logits=0 #
    args.fea_trans ='L2N' # trC_L2N sptC_L2N
    args.bool_autoft=False
    args.optimier='sgd' # adamw, adam, sgd
    args.lr_scheduler='multsteplr' #  consinelr, multsteplr
    
    args.distri='Gaussian' # Gaussian or t_distri (Student)
    args.choleky_alpha=False
    args.reg_param=0.3
    args.epoch=20
    args.metatrain_episode=100
    args.metatest_episode=100
    args.path2features='../data_src/fea_mqda/{:}-{:}-{:}-fea.pkl'.format(domain_net,net,domain_data)
    args.mtr_path=os.path.join(args.datapath+'{:}/train'.format(domain_data))
    args.mva_path=os.path.join(args.datapath+'{:}/val'.format(domain_data))
    args.mte_path=os.path.join(args.datapath+'{:}/test'.format(domain_data))

    args.title='{:}_{:}_{:}_{:}'.format(domain_net,net,args.optimier)
    exp_title='{:}_{:}_{:}_{:}{:}'.format(args.title,str(args.lr),str(args.support_shot),['fea','logits'][args.bool_logits],args.feature_dim)
    # Print Key Hyperparameters by Indicator
    indicator = util.indicator(para_name=['','feature norm','shot','device'],para_value=[exp_title, args.fea_trans, args.support_shot, args.device])
    indicator.print_()
    
    with open(args.path2features,'rb') as f_read:
        presaved_features = pickle.load(f_read)
#     pdb.set_trace()
    classifier = MetaQDA(fea_dim=args.feature_dim, input_process=args.fea_trans, bool_autoft=args.bool_autoft, cholesky_alpha=args.choleky_alpha, prior=args.distri).to(torch.device("cuda"))
            
    if args.fea_trans not in ['UN','L2N','trC_L2N','sptC_L2N']:
            raise ValueError('args.fea_trans should be tr/spt_centre, but your input is', args.fea_trans )

    params=classifier.mqda_parameters()    
    
    if args.optimier == 'adam':
        optimizer  = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optimier == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, nesterov=False)
    else:
        raise ValueError('args.optimier should be adamW, adam, sgd')
        
    if args.lr_scheduler == 'multsteplr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epoch * 0.5), int(args.epoch * 0.9)], gamma=0.1)
    elif args.lr_scheduler == 'consinelr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.metatrain_episode*args.epoch)
    else:
        raise ValueError('args.lr_scheduler should be MultiStepLR or CosineLR')
    cudnn.benchmark = True
    
    log_dir = './log_mqda/{:}'.format(exp_title)
    logger = Logger(str(log_dir), 'setting')
    logger.print('{:} --- args ---'.format(time_string()))
    xargs = vars(args)
    if args.resume:
        resume = torch.load(args.resume)
        classifier=resume['mqda']
        r_args = resume['args']
        vars(r_args)
        pdb.set_trace()
    for key, value in xargs.items():
        logger.print('  [{:18s}] : {:}'.format(key, value))
    logger.print('{:} --- args ---'.format(time_string()))
    ''' ================================= Meta Train & Meta Test ================================= '''
    mtr_folders, mva_folders, mte_folders = dataloder.get_data_folders(args.mtr_path, args.mva_path, args.mte_path)
    if args.mtr_has_val:
        mtr_folders=mtr_folders+mva_folders
    iter_time, timestamp = AverageMeter(), time.time()
    for epoch in range(args.epoch):
        acc, h_, loss = metatrain_epoch(mtr_folders, presaved_features, classifier, optimizer, lr_scheduler)
        info = {'args': args,
            'epoch': epoch,
            'optim' : optimizer.state_dict(),
            'scheduler' : lr_scheduler.state_dict(),
            'mqda' : classifier.state_dict()}
        save_dir = "{:}/iter-{:}.pth".format(log_dir, epoch)
        logger.print('save model {:}'.format(save_dir))
        torch.save(info, save_dir)
        
        if args.lr_scheduler == 'multsteplr':
            lr_scheduler.step()

        iter_time.update(time.time() - timestamp)
        timestamp = time.time()
        time_str  = convert_secs2time(iter_time.avg * (args.epoch - epoch), True)
        logger.print("epoch [{:3d}/{:3d}] episode:{:}, aver_accuracy:{:} h:{:} loss:{:}, still need {:}, {:}".format(epoch+1, args.epoch, args.metatrain_episode, acc, h_, loss, time_str, time_string()))  

    logger.rename('train')
    
def metatrain_epoch(mtr_folders, presaved_features, classifier, optimizer, lr_scheduler):
    accuracies=[]
    for train_episode in range(args.metatrain_episode+1):
        task_train = dataloder.DataLoadTask(mtr_folders, args.support_way, args.support_shot, args.query_shot)
        # Load Features
        support_image_names, support_labels = dataloder.get_name_label(task_train, num_per_class=args.support_shot, split="train").__iter__().next()
        query_image_names, query_labels = dataloder.get_name_label(task_train, num_per_class=args.query_shot, split="test").__iter__().next()
        batch_size = query_labels.shape[0]
        support_features_cpu=util.name2fea(presaved_features, support_image_names, args.datapath, args.bool_logits)
        query_features_cpu=util.name2fea(presaved_features, query_image_names, args.datapath, args.bool_logits)
        
        if args.fea_trans == 'sptC_L2N':
            centre_ = support_features_cpu.mean(0)
        else:
            raise ValueError('args.fea_trans should be tr/spt_centre, but your input is', args.fea_trans )
        # Forward Propagation
        classifier.fit(torch.tensor(support_features_cpu).cuda(), support_labels, torch.tensor(centre_).cuda())
        outputs, cholesky_loss_logitem, cholesky_loss_traceitem, L_diagonal, S_diagonal, S_inverse_diagonal, lower_triu_matrix = classifier.predict(torch.tensor(query_features_cpu).cuda(), torch.tensor(centre_).cuda())

        predicts = outputs.max(dim=1)[1].cpu()
        metatrain_acc, _ = util.accuracy(predicts, query_labels)
        accuracies.append(metatrain_acc)
        # Backward Propagation
        optimizer.zero_grad()

        ce_loss = F.cross_entropy(outputs, query_labels.cuda())
        if args.choleky_alpha:
            loss=ce_loss+ args.choleky_alpha*(cholesky_loss_logitem+cholesky_loss_traceitem)
        else:
            loss=ce_loss
        loss.backward()
        optimizer.step()
        if args.lr_scheduler == 'consinelr':
            lr_scheduler.step()

    accuracy, h_ = util.mean_confidence_interval(accuracies)
    accuracy = round(accuracy * 100, 2)
    h_ = round(h_ * 100, 2)
    loss = round(loss.item(),2)
    return accuracy, h_, loss
    
def metatest_epoch(mte_folders, presaved_features, classifier, epoch):
    print('Meta Testing...\tepoch:{:}/{:} episode:{:}'.format(epoch+1, args.epoch ,args.metatest_episode))
    accuracies = []
    for unused_i in range(args.metatest_episode):
        del unused_i
        task_test = dataloder.DataLoadTask(mte_folders, args.support_way, args.support_shot, args.query_shot)
        # Load Features
        support_image_names, support_labels = dataloder.get_name_label(task_test, num_per_class=args.support_shot, split="train").__iter__().next()
        query_image_names, query_labels = dataloder.get_name_label(task_test, num_per_class=args.query_shot, split="test").__iter__().next()
        batch_size = query_labels.shape[0]
        support_features_cpu=util.name2fea(presaved_features, support_image_names, args.datapath, args.bool_logits)
        query_features_cpu=util.name2fea(presaved_features, query_image_names, args.datapath, args.bool_logits)
        if args.fea_trans == 'trC_L2N':
            centre_ = presaved_features['trCentre'][args.bool_logits]
        elif args.fea_trans == 'sptC_L2N':
            centre_ = support_features_cpu.mean(0)
        else:
            raise ValueError('args.fea_trans should be tr/spt_centre, but your input is', args.fea_trans )
        # Inference
        classifier.fit(torch.tensor(support_features_cpu).cuda(), support_labels, torch.tensor(centre_).cuda())
        outputs = classifier.predict(torch.tensor(query_features_cpu).cuda(),torch.tensor(centre_).cuda())[0].detach().cpu()
        # Record Accuracy
        ce_loss = F.cross_entropy(outputs, query_labels)
        loss=ce_loss
        predicts = outputs.max(dim=1)[1]
        metatest_acc, _ =util.accuracy(predicts, query_labels)
        accuracies.append(metatest_acc)
    # confidence_interval for accuray record
    accuracy, h_ = util.mean_confidence_interval(accuracies)
    accuracy = round(accuracy * 100, 2)
    h_ = round(h_ * 100, 2)
    loss = round(ce_loss.item(),2)
    print("meta_test aver_accuracy:", accuracy, "h:", h_, 'loss:', loss)
    print()
    return accuracy, h_, loss

if __name__=='__main__':
    main()
