# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:28:46 2022

@author: m.dorosti
"""

with open('C:/Users/paya8/Desktop/jobs/a.txt','r' ,encoding='utf-8') as file:
    path=file.read().splitlines()
        
path=path[0]    
print(path)