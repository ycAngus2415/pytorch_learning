import torch

netG_state_dic=torch.load('netG_epoch_13.pth')

netD_state_dic=torch.load('netD_epoch_13.pth')
b=netG_state_dic.keys()
dict={'key':4,'key3':5}


print(netD_state_dic['main.0.weight'])
print(netD_state_dic.keys())
