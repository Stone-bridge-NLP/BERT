import numpy as np
import pandas as pd
import torch

def dfRandomSample(dataframe, frac, return_loss_coef = True, verbose = True):
  Genre = dataframe['Genre'].value_counts(ascending=True).index
  max_portion = np.array(dataframe['Genre'].value_counts(ascending=True).values, dtype=np.long)
  
  portion = []
  leftover = int(len(dataframe)*frac)
  for i, v in enumerate(max_portion):
    if v > leftover//(len(Genre)-i):
      portion.append(leftover//(len(Genre)-i))
    else:
      portion.append(v)
    leftover -= portion[-1]
  
  if verbose:
    print('='*20)

  reduced = []
  for i in range(len(Genre)):
    df = dataframe[dataframe['Genre'] == Genre[i]]
    reduced.append(df.sample(portion[i],random_state=31415))
    if verbose:
      print(f'{Genre[i]}\t sample ratio\t {len(df)}:{len(reduced[-1])}')

  reduced = pd.concat(reduced).reset_index().drop(columns='index')
  if verbose:
    print(f'Total size : {len(reduced)}', end='\n\n')

  if not return_loss_coef:
    return reduced

  else:
    class_size = reduced['Genre'].value_counts()
    names=class_size.index
    ratio = np.array(class_size.values)
    return reduced, {name:coef for name, coef in zip(names,(1/ratio)*(len(reduced)/10))}

def save_checkpoint(current_epoch, model, PATH):
  torch.save({
      'current_epoch':current_epoch,
      'model':model.state_dict()
  }, PATH)