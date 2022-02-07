from tkinter import Variable
import numpy as np
import torch
from torch import Tensor, conv2d, nn
from torch.nn import functional as F

#from layers import Conv1d, Res1d, Linear, LinearRes, Null

class HOME(nn.Module):
    def __init__(self,config=None):
        super(HOME, self).__init__()
        self.img_conv = IMG_conv_layer()
        self.interact = Interaction_layer()
        self.decode = Decoder()
        self.linear = nn.Linear(50,288*288)
        self.softmax = nn.Softmax()
    def forward(self, x):
        context_encode = self.img_conv(x['img'])
        social_encode = self.interact(x['history']).permute(0,3,1,2)
        encode = torch.cat((context_encode,social_encode),1)
        decode = self.decode(encode)
        return decode
        #x = self.linear(x['img'][:,0,0,0:50])
        #x = self.softmax(x).reshape(-1,288,288)
        #return x

class IMG_conv_layer(nn.Module):
    def __init__(self,config=None):
        super(IMG_conv_layer, self).__init__()
        #224->224->112->56->28->14
        self.net = nn.Sequential(
            nn.Conv2d(45,32,kernel_size=5,padding=2,stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,kernel_size=6,padding=2,stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=6,padding=2,stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128,256,kernel_size=6,padding=2,stride=2),
            nn.ReLU(inplace=True), 
            nn.Conv2d(256,512,kernel_size=6,padding=2,stride=2),
            nn.ReLU(inplace=True),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.net(x)

class Interaction_layer(nn.Module):
    def __init__(self,config=None):
        super(Interaction_layer, self).__init__()
        #bs,3,H
        #bs,64,H
        #bs,channel,length
        #output:seq_length,batch_size,hidden_size=128
        
        # x * 128 -> 1 * 128 -> Linear
        self.agent_encode_conv = nn.Sequential(
            nn.Conv1d(3,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True)
        )
        self.agent_encode_lstm = nn.LSTM(input_size=64,hidden_size=128)

        self.other_encode_conv = nn.Sequential(
            nn.Conv1d(3,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True)
        )
        self.other_encode_lstm = nn.LSTM(input_size=64,hidden_size=128)
        #self.other_encode 
        self.Attlayer = Attention(128, 128)
        self.layernorm = nn.LayerNorm(128)
        self.linear = nn.Linear(128,128)

    def forward(self,x:Tensor):
        # x: bs, N+1, H, 3
        N = x.shape[1]
        all_vehicle_out = []

        #print(x[:,0,:,:].shape)
        agent = self.agent_encode_conv(x[:,0,:,:])
        agent = agent.permute(2,0,1).contiguous()
        _,(agent_hn,__)  = self.agent_encode_lstm(agent)
        agent_out = agent_hn.squeeze(0)

        all_vehicle_out.append(agent_out)

        for other_id in range(1,N):
            other = self.other_encode_conv(x[:,0,:,:])
            other = other.permute(2,0,1).contiguous()
            _,(other_hn,__)  = self.other_encode_lstm(agent)
            other_out = other_hn.squeeze(0)
            all_vehicle_out.append(agent_out)
        
        all_vehicle_out = torch.stack(all_vehicle_out,dim=1)

        res = self.Attlayer(all_vehicle_out)
        res = self.layernorm(res)
        res = self.linear(res)
        res = res.unsqueeze(1).unsqueeze(1)
        res = res.repeat(1,14,14,1)
        #output, (hn,cn) = self.agent_encode(x[:,0,:,:])
        #for 
        return res

class Decoder(nn.Module):
    #in:14,14,640
    def __init__(self):
        super(Decoder, self).__init__()
        #16 - he + 2 * p + 1 = 14
        self.tr_conv1 = nn.ConvTranspose2d(640,512,kernel_size=3,padding=0,stride=1)
        self.relu1 =  nn.ReLU(inplace=True)
        self.tr_conv2 = nn.ConvTranspose2d(512,512,kernel_size=3,padding=0,stride=1)
        self.relu2 =  nn.ReLU(inplace=True)

        self.conv_decode = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256,128,kernel_size=4,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,64,kernel_size=4,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64,32,kernel_size=4,padding=1,stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32,1,kernel_size=4,padding=1,stride=2),
            nn.Sigmoid()
        )

        #self.softmax = nn.Softmax()
    def forward(self, x): 
        x = self.tr_conv1(x)
        x = self.relu1(x)
        x = self.tr_conv2(x)
        x = self.relu2(x)
        #x = 
        x = self.conv_decode(x)
        x = x.squeeze(1)
        sz = x.shape[2]
        batchsize = x.shape[0]
        #print(x)
        #x = x.squeeze(1)
        #x = self.softmax(x.reshape(batchsize,-1)).reshape(batchsize,sz,sz)
        #print(x.shape)
    
        return x

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, y_predict:Tensor, gt_pos:np.ndarray) -> Tensor:
        #print("!")
        loss = Tensor()
        loss = 0
        #for batch_id in range(
        device = y_predict.device
        batch_size = y_predict.shape[0]
        img_size = y_predict.shape[1]
        gt_pos = Tensor(gt_pos).to(device).long()
        #print(batch_size)
        cnt = 0
        #Tensor 
        print(y_predict.shape)
        for id in range(batch_size):
            for i in range(-3,3):
                for j in range(-3,3):
                    #if i != 0 and j != 0:
                    #    continue
                    pos = (gt_pos[id] + Tensor([i,j]).to(device)).long()
                    #print(pos)
                    #print(img_size)
                    if pos[0] < 0 or pos[0] >= img_size or pos[1] < 0 or pos[1] >= img_size:
                        continue
                    y_hat = y_predict[id][tuple(pos)]
                    #print(y_hat.shape)
                    y = torch.exp( (-torch.norm((pos - gt_pos).float())**2) / 5 )
                    if y == 1:
                        #print(y_hat)
                        loss += (-torch.log(y_hat)*(y_hat-y)**2)
                    else:
                        loss += (-torch.log(1-y_hat)*((1-y)**4)*(y_hat-y)**2)
                    cnt += 1
        
        #return Tensor(np.mean(loss)).to(y_predict.device)
        #print(loss)
        #loss_ = torch.mean(Tensor(loss))
        #loss_ = loss_.requires_grad_()
        return loss / cnt
        #notice:device

class Attention(nn.Module):
    def __init__(self, feature_size, num_hiddens):
        
        super(Attention, self).__init__()
        self.W1 = nn.Linear(feature_size * 2, num_hiddens, bias=False)
        self.W2 = nn.Linear(num_hiddens, feature_size, bias=False)
        self.W0 = nn.Linear(feature_size, feature_size, bias=False)

    def forward(self, x):
        #queries: bs, num_of_queries, feature 
        #queries, keys = self.W_q(queries), self.W_k(keys)
        #print(queries.unsqueeze(2).shape)
        #print(keys.unsqueeze(1).shape)
        #features = queries.unsqueeze(2) + keys.unsqueeze(1)
        #return 1   
        N = x.shape[1]
        
        res = self.W0(x[:,0,:])
        for i in range(1,N):
            interact = torch.cat((x[:,0,:],x[:,i,:]),1)
            interact = self.W1(interact)
            interact = torch.tanh(interact)
            interact = self.W2(interact)
            res += interact
      
        return res
'''
class BasicBlock(nn.Module):
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes, stride=1):
        super(Batorch重复tensorsicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out
'''

if __name__ == '__main__':
    #net = IMG_conv_layer()
    #a = torch.randn((1,45,224,224))
    #print(net(a).shape)
    #net = Interaction_layer()
    #x = torch.rand((32,5,3,20))
    #print(net(x).shape)
    #net = Decoder()
    #net = net.cuda()
    #x = torch.rand((32,640,14,14)).cuda()
    #print(net(x).shape)
    #print(x.device)
    #t = torch.rand((1,288,288)).cuda()
    
    y = np.zeros((32,2))
    if False:
        x[0,150,150] = 1
        x[0,149,150] = 0
        x[0,150,151] = 0
        x[0,150,149] = 0
        x[0,151,150] = 0
    #l = loss()
    #print(l(x,y))
    #loss
    net = HOME()
    net = net.cuda()
    x = dict()
    x['img'] = torch.randn(32,45,224,224).cuda()
    x['history'] = torch.randn(32,100,3,20).cuda()
    #x['num_of_agents'] = [5] * 32
    l = loss().cuda()
    t = l(net(x),y)
    
    #t.backward()
