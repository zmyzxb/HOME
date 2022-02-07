from calendar import EPOCH
from home import * 
from preprocess import * 
import torch

config_file = open('config.json')
config = json.load(config_file)

dataset = ArgoDataset()
dl = DataLoader(dataset,batch_size = 1, collate_fn = collate)
'''
for idx, (dt, label) in enumerate(dl):
    print(label.shape)
    print(dt['img'].shape)
    print(dt['history'].shape)
    print(dt['num_of_agents'])
'''
# loop over the dataset multiple times
net = HOME(config)
net = net.cuda()
device = torch.device('cuda')
optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
l = loss().cuda()
net.train()
losses = []
for epoch in range(config["train_epoch"]):
    running_loss = 0.0
    for i, data in enumerate(dl, 0):
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(torch.max(torch.max(torch.max(outputs))))
        loss = l(outputs, labels)
        #print(outputs[0,145,140:150])
        print(loss)
        loss.backward()
        optimizer.step()
        #for name, parms in net.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
        #        ' -->grad_value:',parms.grad)
        running_loss += loss.item()
    losses.append(running_loss)
    print('Loss: {}'.format(running_loss))
print(running_loss)
print('Finished Training')