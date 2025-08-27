import numpy as np
import torch

bikes_numpy = np.loadtxt("C:\\Users\\48697\\Desktop\\ai\\Ai-learning\\Practice\\Year\\hour-fixed.csv",
                         dtype=np.float32,
                         delimiter=",",
                         skiprows=1,
                         converters={1: lambda x: float(x[8:10])})
bikes = torch.from_numpy(bikes_numpy) 

daily_bikes = bikes.view(-1, 24, bikes.shape[1]) 
daily_bikes = daily_bikes.transpose(1, 2) 
first_day = bikes[:24].long() 
weather_onehot = torch.zeros(first_day.shape[0], 4) 
daily_weather_onehot = torch.zeros(daily_bikes.shape[0], 4, daily_bikes.shape[2])
daily_weather_onehot.scatter_(1, daily_bikes[:,9,:].long().unsqueeze(1) - 1, 1.0) 