import os
os.environ['OPENBLAS_NUM_THREADS'] = '0,1'
import time
import torch
import torch.nn as nn
from torch import optim
import pdb
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data import MyDataset_train, MyDataset_test

from model import speech_music_classify
from tensorboardX import SummaryWriter
import tensorflow as tf

if(__name__ == '__main__'):
    torch.manual_seed(55)
    # torch.cuda.manual_seed_all(55)
    opt = __import__('options')


def data_from_train():
    dataset = MyDataset_train()
    print('num_data:{}'.format(len(dataset.data)))
    
    loader = DataLoader(dataset, 
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True,batch_size= 64)   
    return (dataset, loader)

def data_from_test():
    dataset = MyDataset_test()
    print('num_data:{}'.format(len(dataset.data)))
    
    loader = DataLoader(dataset, 
        num_workers=opt.num_workers,
        drop_last=False,
        shuffle=True,batch_size=64)
    return (dataset, loader)

if(__name__ == '__main__'):
    # model = speech_music_classify().cuda()
    model = speech_music_classify()

    writer = SummaryWriter()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        # print(model_dict.keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)       

    (train_dataset, train_loader) = data_from_train()
    (test_dataset, test_loader) = data_from_test()
    #(tst_dataset, tst_loader) = data_from_opt(opt.tst_txt_path, 'test')
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
             lr=opt.lr,
             weight_decay=0)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    iteration = 0
    m = 0
    for epoch in range(opt.max_epoch):
        start_time = time.time()
        exp_lr_scheduler.step()

        for (i, batch) in enumerate(train_loader):
            # (inputs, label) = batch['inputs'].cuda(), batch['label'].cuda()
            (inputs, label) = batch['inputs'], batch['label']
            #print(inputs.size())
            inputs = inputs.view(opt.batch_size, 39, -1)
            #print(inputs.size())
            logit = model(inputs)
            loss = criterion(logit, label)
            optimizer.zero_grad()   
            iteration += 1
            loss.backward()
            optimizer.step()
            tot_iter = epoch*len(train_loader)+i
            writer.add_scalar('data/CE_loss', loss.detach().cpu().numpy(), iteration)
            train_loss = loss.item()
            #if i%500==0:
            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))
            n = int(len(train_dataset.data) / opt.batch_size)
            if (i+1)==n:
                break
            
        if (iteration % 10) == 0:
                with torch.no_grad():
                    corrects = 0
                    for (idx,batch) in enumerate(test_loader):
                        #(inputs, label) = batch['inputs'].cuda(), batch['label'].cuda()
                        (inputs, label) = batch['inputs'], batch['label']
                        inputs = inputs.view(opt.batch_size, 39, -1)
                        logit = model(inputs)
                        _, predict_txt = torch.max(F.softmax(logit, dim=1).data, 1)
                        corrects += torch.sum(predict_txt == label)
                        n = int(len(test_dataset.data) / opt.batch_size)
                        if (idx+1)==n:
                            break
                    m += 1
                    acc = corrects.item() / (n*opt.batch_size)
                    writer.add_scalar('acc/accuracy', acc, m)
                    print('the acc is: ', acc)
                savename = os.path.join(opt.save_dir1, 'iteration_{}_epoch_{}_acc_{}.pt'.format(iteration, epoch, acc))
                savepath = os.path.split(savename)[0]
                if(not os.path.exists(savepath)): os.makedirs(savepath)
                torch.save(model.state_dict(), savename)
