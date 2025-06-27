import torch
import json

with open("a.json",'r',encoding='utf-8') as f:
    a =json.load(f)['a']

with open("b.json",'r',encoding='utf-8') as f:
    b=json.load(f)['a']

with open("a+b.json",'r',encoding='utf-8') as f:
    ab=json.load(f)['a']

with open("b+b.json",'r',encoding='utf-8') as f:
    bb=json.load(f)['a']   

with open("b+a.json",'r',encoding='utf-8') as f:
    ba=json.load(f)['a+b'] 

for i in ba:
    idx, rank = i
    if rank == 0:
        if i not in a and i not in bb and i not in ab and i not in b:
            print(idx,rank)