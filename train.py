import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import numpy as np
import torch.utils.data as DataSet
from TFCTL_network import TFCTL
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix
from dataloader_calibration import *
from tripleloss import TripletLoss
def get_metrics_fixed(SR,GT):

    prediction_bin=torch.tensor(SR)
    label = torch.tensor(GT)
    true_positives = torch.sum(torch.logical_and(label == 1, prediction_bin == 1))
    false_positives = torch.sum(torch.logical_and(label == 0, prediction_bin == 1))
    false_negatives = torch.sum(torch.logical_and(label == 1, prediction_bin == 0))
    true_negatives = torch.sum(torch.logical_and(label == 0, prediction_bin == 0))
    Precision = true_positives / (true_positives + false_positives + 1e-10)
    Recall = true_positives / (true_positives + false_negatives + 1e-10)
    F1score = 2 * (Precision * Recall) / (Precision + Recall)
    return F1score,Precision, Recall

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        # nn.init.normal_(m.weight, mean=0,  std=np.sqrt(2 / 256))

def normalize(data_set):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    # max=data_set.max(axis=(0, 1, 2, 3))
    #
    # min=data_set.min(axis=(0, 1, 2, 3))
    mean = np.mean(data_set, axis=(0, 1, 2, 3))
    std = np.std(data_set, axis=(0, 1, 2, 3))
    data_set1 = (data_set - mean) / (std + 1e-7)
    return data_set1

epochs=50
BATCH_SIZE=32
LR = 1e-4
DOWNLOAD_MNIST = True
if_use_gpu = True
method="train"

model_name="TFCTL"
model_dir='./checkpoint/' + model_name
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
print(model_name)
train_ds = wav_dataloader(r"/home/dell/dataset/DAIC_wav_npy",30)
test_ds = wav_dataloader(r"/home/dell/dataset/CMDC_wav_npy",30)
train_loader = DataSet.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataSet.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
vali_loader = DataSet.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loss_function = nn.CrossEntropyLoss()
tr_loss=TripletLoss()
if method=="train":
    # resnet = models.resnet18()
    model=TFCTL()
    # print(cnn)
    model.apply(init_weights)
    if if_use_gpu:
        resnet = model.cuda()
    old_acc=0
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    min_loss_val = 0  # 任取一个大数
    start_time = time.time()
    for epoch in range(1,epochs+1):
        start = time.time()
        F1_list = []
        Recall_list = []
        Precision_list = []
        train_loss_list = []
        full_preds = []
        full_gts = []
        model=model.train()
        len_dataloader = max(len(train_loader), len(test_loader))
        data_source_iter = iter(train_loader)
        data_target_iter = iter(test_loader)
        step=0
        while step <len_dataloader:
            # for step, (s_img,s_label) in enumerate(train_loader):
            p = float(step + epoch * len_dataloader) / epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            if step % len(train_loader) == 0:
                data_source_iter = iter(train_loader)
            s_x,s_y,sp,s_domain,p_data,n_data = data_source_iter.next()
            s_x = Variable(s_x, requires_grad=False).float()
            s_y = Variable(s_y, requires_grad=False)
            s_domain= Variable(s_domain, requires_grad=False)
            if if_use_gpu:
                s_x = s_x.cuda()
                s_y = s_y.cuda()
                s_domain = s_domain.cuda()
            output, vec,domain= model(s_x, alpha=alpha)
            loss_cls = loss_function(output, s_y)
            loss_domain_s = loss_function(domain, s_domain)

            if step%len(test_loader)==0:
                data_target_iter = iter(test_loader)
            t_x, t_y, t_sp, t_domain,_,_ = data_target_iter.next()
            t_x = Variable(t_x, requires_grad=False).float()
            t_y = Variable(t_y, requires_grad=False)
            t_domain = Variable(t_domain, requires_grad=False)
            if if_use_gpu:
                t_x = t_x.cuda()
                t_y = t_y.cuda()
                t_domain = t_domain.cuda()
            output, _, domain = model(t_x, alpha=alpha)
            loss_domain_t = loss_function(domain, t_domain)
            _, p_e, _= model(p_data, alpha=alpha)
            _, n_e, _ = model(n_data, alpha=alpha)
            loss_triple=tr_loss(vec,p_e,n_e)
            loss=loss_domain_s+loss_domain_t+loss_cls+loss_triple
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predictions = np.argmax(output.detach().cpu().numpy(), axis=1)
            step=step+1

        if epoch % 1 == 0 or epoch == epochs - 1:
            # model=model.eval()
            test_loss_list = []
            full_preds = []
            full_gts = []
            embedding = []
            full_scores=[]
            for x,y,sp,domain in test_loader:
                b_x = Variable(x, requires_grad=False).float()
                b_y = Variable(y, requires_grad=False)
                #print(b_x.shape)
                if if_use_gpu:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                # output,r_vec = resnet(b_x)
                output,vec,_ = model(b_x,alpha)
                loss = loss_function(output, b_y)
                predictions = np.argmax(output.detach().cpu().numpy(), axis=1)
                scores = np.max(output.detach().cpu().numpy(), axis=1)
                test_loss_list.append(loss.item())
                for index, pred in enumerate(predictions):
                    full_preds.append(pred)
                    full_scores.append(scores[index])
                for lab in b_y.detach().cpu().numpy():
                    full_gts.append(lab)

            mean_acc = accuracy_score(full_gts, full_preds)
            mean_loss = np.mean(np.asarray(test_loss_list))
            et = time.time() - start_time
            et = str(datetime.timedelta(seconds=et))[:-7]
            F1, Precision, Recall, AUC = get_metrics_fixed(full_preds, full_gts, full_scores)
            #print(full_preds)
            print(et, 'Epoch:', epoch, '|testloss:%.4f' % mean_loss, '|test accuracy:%.4f' % mean_acc,
                  '|test F1:%.4f' % F1, '|test Precision:%.4f' % Precision,
                  '|test Recall:%.4f' % Recall, '|test AUC:%.4f' % AUC)
            if old_acc>mean_acc:
                old_acc=mean_acc
                model_save_path = os.path.join('checkpoint/' + model_name)
                state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state_dict, model_save_path)
# resnet =models.resnet18()
model=TFCTL()
model.load_state_dict(torch.load(model_dir)["model"])
epoch=torch.load(model_dir)["epoch"]
# cnn2 = torch.load("./model/" + target_name + "/" + database + "test_model" + str(round) + ".pkl")
print("test mode")
if if_use_gpu:
    model = model.cuda()
#model = model.eval()
test_loss_list = []
full_preds = []
full_gts = []
embedding=[]
acc=[0,0,0,0,0]
F1_list = []
Recall_list=[]
Precision_list=[]
sp_list=[]
full_scores=[]
for step,(x,y,sp,domain) in enumerate(vali_loader):
    b_x = Variable(x, requires_grad=False).float()
    b_y = Variable(y, requires_grad=False)
    if if_use_gpu:
        b_x = b_x.cuda()
        b_y = b_y.cuda()
    # output,r_vec = resnet(b_x)
    output, vec,_ = model(b_x,0)
    predictions = np.argmax(output.detach().cpu().numpy(), axis=1)
    scores = np.max(output.detach().cpu().numpy(), axis=1)
    loss = loss_function(output, b_y)
    test_loss_list.append(loss.item())
    # accuracy = sum(pred_y == b_y) / b_y.size(0)
    for index, pred in enumerate(predictions):
        full_preds.append(pred)
        full_scores.append(scores[index])
    for lab in b_y.detach().cpu().numpy():
        full_gts.append(lab)
    for sp_slice in sp:
        sp_list.append(sp_slice)
F1, Precision, Recall,AUC = get_metrics_fixed(full_preds, full_gts,full_scores)
mean_acc = accuracy_score(full_gts, full_preds)
mean_loss = np.mean(np.asarray(test_loss_list))
print('test accuracy:%.4f' % mean_acc,
      'test F1:%.4f' % F1, '|Fixed test Precision:%.4f' % Precision,
      'test Recall:%.4f' % Recall,'|test AUC:%.4f' % AUC)

