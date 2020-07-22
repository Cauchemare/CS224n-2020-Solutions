#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
Usage:
  cnn.py view
  cnn.py value
  cnn.py -h
"""

from docopt import docopt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self,embed_size,out_channels,kernel_size=5,padding=1):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
        super().__init__()
        self.conv1d= nn.Conv1d(in_channels= embed_size,out_channels= out_channels,kernel_size=kernel_size,padding=padding)


    def  forward(self,x_reshaped):
        '''
        @params x_reshaped(Tensor(batch_size,embed_char,m_word) ) batch of words
        @return x_conv (Tensor (b,out_channels))
        '''
        b,embed_char,m_word=  x_reshaped.size()
        hidden1  = self.conv1d(x_reshaped)      #(b,out_channels,(m_word-k)/padding+1 )
        x_conv =torch.max(nn.functional.relu(   hidden1 ),dim=2).values #(b,out_channels)
        return  x_conv
    
    ### END YOUR CODE


if __name__ == '__main__':
    args = docopt(__doc__)
    seed = 2020
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed // 2)
    
    x = torch.tensor([[[1., 1., 1., 1.],
                  [-2, -2, -2., -2.]],
                 [[2, 2, 1, 1],
                  [0.5, 0.5, 0, 0]]], dtype=torch.float32)
    print("input tensor shape:  ", x.size())
    x = x.permute(0, 2, 1).contiguous()
    model = CNN(x.size()[-2], 3, kernel_size=2)
    if args['view']:
        print("model's parameter print...")
        for p in model.parameters():
            print(p)
    elif args['value']:
        print("value confirmation...")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)
        x_conv = model(x)
        print("input:\n{}\nsize: {}".format(x, x.size()))
        print("output:\n{}\nsize: {}".format(x_conv, x_conv.size()))