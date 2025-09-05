import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import empyrical as ep
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerGPT2Model
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# class
class CustomDataset(Dataset):
    def __init__(self, *args):
        self.data = args
    def __len__(self):
        return len(self.data[0])
    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.data)

class DataPreprocess:
    def __init__(self):
        self.max_length = max_len
        self.max_ep_len = 4096
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99

    def split_data(self, state_subset, action_subset, return_subset):
        s, a, t, R, mask = [], [], [], [], []
        for start_idx in range(len(state_subset) - (self.max_length-1)):
            state_piece = state_subset[start_idx:start_idx + self.max_length].reshape(1, -1, self.state_size)
            seq_length = state_piece.shape[1]
            min_val = np.min(state_piece, axis=1)
            max_val = np.max(state_piece, axis=1)
            state_norm = (state_piece-min_val) / (max_val-min_val)
            s.append(state_norm)

            action_piece = action_subset[start_idx:start_idx + self.max_length].reshape(1, -1, self.action_size)
            a.append(action_piece)

            timesteps = np.arange(self.max_length).reshape(1, -1)
            t.append(timesteps)

            returns_to_go_piece = count_return_to_go(return_subset[start_idx:start_idx + self.max_length], self.gamma).reshape(1, -1, 1)
            R.append(returns_to_go_piece)

            mask.append(np.ones((1, seq_length)))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        rtg = torch.from_numpy(np.concatenate(R, axis=0)).to(dtype=torch.float32, device=device)
        t = torch.from_numpy(np.concatenate(t, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, t, rtg, mask

class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        config = DecisionTransformerConfig(state_dim = self.state_size,
                                           act_dim = self.action_size,
                                           hidden_size = hidden_size,
                                           max_ep_len = 4096)
        self.transformer = DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, self.hidden_size)
        self.embed_state = nn.Linear(self.state_size, self.hidden_size)
        self.embed_action = nn.Linear(self.action_size, self.hidden_size)
        self.embed_return = nn.Linear(1, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)

        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_size)
        self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, config.act_dim),
            nn.Tanh()
            #nn.Softmax(dim=-1)
        )

    def forward(self, state, action, return_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = state.shape[0], state.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(state.device)

        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(state) + time_embeddings
        action_embeddings = self.embed_action(action) + time_embeddings
        returns_embeddings = self.embed_return(return_to_go) + time_embeddings

        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)
        stacked_attention_mask = torch.stack((attention_mask, attention_mask, attention_mask), dim=1).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs,
                                               attention_mask = stacked_attention_mask)

        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        return_pred = self.predict_return(x[:, 2])
        state_pred = self.predict_state(x[:, 2])
        action_pred = self.predict_action(x[:, 1])

        return state_pred, action_pred, return_pred

    def loss_fn(self, action_pred, action_target):
        action_pred = action_pred.reshape(-1, self.action_size)
        action_pred = F.softmax(action_pred, dim=1)
        action_target = action_target.reshape(-1, self.action_size)
        loss = torch.mean((action_pred - action_target) ** 2)

        return loss

    def get_action(self, state, action, return_to_go, timesteps):
        state = state.reshape(1, -1, self.state_size)
        action = action.reshape(1, -1, self.action_size)
        return_to_go = return_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        attention_mask = None

        _, action_pred, return_pred = self.forward(state, action, return_to_go, timesteps, attention_mask=attention_mask)

        return action_pred[0, -1]

# function
def count_return_to_go(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def compute_dr(close_today, close_tomorrow, weight_today):
    trans_cost = 0.001
    portfolio_value = 1000000

    share_today = np.floor(weight_today.cpu().numpy() * portfolio_value / close_today)
    cash_today = portfolio_value - sum(share_today * close_today) # share_hold

    if trans_cost > 0:
        trans_cost_today = np.sum(share_today * close_today) * trans_cost

    portfolio_value_tomorrow = sum(share_today * close_tomorrow) + cash_today - trans_cost_today
    dr = (portfolio_value_tomorrow - portfolio_value) / portfolio_value

    return dr

def train(year):
    model.load_state_dict(torch.load(f'./model_weights/original.pth'))
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    for epoch in range(epochs):
        total_sample = 0
        total_loss = 0
        for data in tqdm(train_dataloader, total=len(train_dataloader)):
            optimizer.zero_grad()
            state_batch, action_batch, timesteps_batch, return_to_go_batch, mask_batch = data
            _, action_pred, _ = model(state_batch, action_batch, return_to_go_batch, timesteps_batch, attention_mask=mask_batch)
            loss = model.loss_fn(action_pred, action_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            total_sample += state_batch.shape[0]
            total_loss += loss.item()
        epoch_loss = total_loss / total_sample
        print(f'Epoch [{epoch+1}/{epochs}] loss = {epoch_loss}')
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), f'./model_weights/{year}_len{max_len}.pth')

def test(year):
    model.load_state_dict(torch.load(f'./model_weights/{year}_len{max_len}.pth'))
    model.eval()
    dt_weights = []
    target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
    state_batch = torch.zeros((0, state_size), device=device, dtype=torch.float32)
    action_batch = torch.zeros((1, action_size), device=device, dtype=torch.float32)
    timesteps_batch = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(state_test[:-1])):
            state_batch = torch.cat([state_batch, torch.from_numpy(data).reshape(1, state_size).to(device=device, dtype=torch.float32)], dim=0)
            if state_batch.shape[0] > max_len:
                state_batch = state_batch[1:]
            min_val = torch.min(state_batch, axis=0).values
            max_val = torch.max(state_batch, axis=0).values
            state_norm = (state_batch-min_val) / (max_val-min_val)
            for i in range(len(max_val)):
                if max_val[i] == min_val[i]:
                    state_norm[:, i] = state_batch[:, i] - min_val[i]
            action_pred = model.get_action(state_norm, action_batch, target_return, timesteps_batch)
            action_pred = F.softmax(action_pred)
            if idx >= max_len-1:
                dt_weights.append(action_pred.tolist())

            action_batch = torch.cat([action_batch, action_pred.reshape(1, action_size)], dim=0)
            if action_batch.shape[0] > max_len:
                action_batch = action_batch[1:]
            reward_pred = compute_dr(data[[0, 3, 6]], state_test[idx+1][[0, 3, 6]], action_pred)
            next_target_return = target_return[0, -1] - reward_pred
            target_return = torch.cat([target_return, next_target_return.reshape(1, 1)], dim=1)
            if target_return.shape[1] > max_len:
                target_return = target_return[:, 1:]
            if timesteps_batch.shape[1] < max_len:
                timesteps_batch = torch.cat([timesteps_batch, torch.ones((1, 1), device=device, dtype=torch.long) * (idx+1)], dim=1)

    np.save(f'./act_weights/dt_weights_{year}_len{max_len}_{TARGET_RETURN}', np.array(dt_weights))

if __name__ == '__main__':
    # device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(device)

    # seed
    torch_seed = 42
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load and split data
    action_data = np.load('/home/minyu/Financial_RL/dt/new_expert/action.npy')
    action_data1 = np.load('/home/minyu/Financial_RL/dt/new_expert/action1.npy')
    feat_data = np.load('/home/minyu/Financial_RL/all_feat.npy')
    return_data = np.load('/home/minyu/Financial_RL/dt/new_expert/daily_return.npy', allow_pickle=True).astype(float)
    return_data1 = np.load('/home/minyu/Financial_RL/dt/new_expert/daily_return1.npy', allow_pickle=True).astype(float)
    date_data = np.load('/home/minyu/Financial_RL/all_date.npy', allow_pickle=True)

    return_data[np.isnan(return_data)] = 0
    return_data1[np.isnan(return_data1)] = 0

    start_year = 2005
    end_year = 2023

    for idx, date in enumerate(date_data):
        if str(start_year) in date:
            cut_idx = idx
            date_data = date_data[cut_idx:]
            feat_data = feat_data[cut_idx:]
            return_data = return_data[cut_idx:]
            return_data1 = return_data1[cut_idx:]
            break

    year_list = list(range(start_year, end_year))
    year_start_idx = {}
    for y in year_list:
        for idx, date in enumerate(date_data):
            if str(y) in date:
                year_start_idx[y] = idx
                break

    # hyperparameters
    state_size = feat_data.shape[1]
    action_size = action_data.shape[1]
    max_len = 20
    batch_size = 64
    hidden_size = 128
    epochs = 200
    TARGET_RETURN = 1

    for test_year in range(start_year+1, end_year):
        print(test_year)
        train_len = year_start_idx[test_year]
        if test_year != 2022:
            date_train, date_test = date_data[:train_len], date_data[train_len-(max_len-1):year_start_idx[test_year+1]]
            state_train, state_test, action_train, action_test, return_train, return_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(max_len-1):year_start_idx[test_year+1]], action_data[:train_len], action_data[train_len-(max_len-1):year_start_idx[test_year+1]], return_data[:train_len], return_data[train_len-(max_len-1):year_start_idx[test_year+1]]
            state1_train, state1_test, action1_train, action1_test, return1_train, return1_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(max_len-1):year_start_idx[test_year+1]], action_data1[:train_len], action_data1[train_len-(max_len-1):year_start_idx[test_year+1]], return_data1[:train_len], return_data1[train_len-(max_len-1):year_start_idx[test_year+1]]
        else:
            date_train, date_test = date_data[:train_len], date_data[train_len-(max_len-1):]
            state_train, state_test, action_train, action_test, return_train, return_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(max_len-1):], action_data[:train_len], action_data[train_len-(max_len-1):], return_data[:train_len], return_data[train_len-(max_len-1):]
            state1_train, state1_test, action1_train, action1_test, return1_train, return1_test = feat_data[:-1][:train_len], feat_data[:-1][train_len-(max_len-1):], action_data1[:train_len], action_data1[train_len-(max_len-1):], return_data1[:train_len], return_data1[train_len-(max_len-1):]

        # data preprocess
        dp = DataPreprocess()
        s_train, a_train, t_train, rtg_train, mask_train = dp.split_data(state_train, action_train, return_train)
        s1_train, a1_train, t1_train, rtg1_train, mask1_train = dp.split_data(state1_train, action1_train, return1_train)
        s_train, a_train, t_train, rtg_train, mask_train = torch.cat((s_train, s1_train), axis=0), torch.cat((a_train, a1_train), axis=0), torch.cat((t_train, t1_train), axis=0), torch.cat((rtg_train, rtg1_train), axis=0), torch.cat((mask_train, mask1_train), axis=0)
        train_dataset = CustomDataset(s_train, a_train, t_train, rtg_train, mask_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = DecisionTransformer().to(device)
        if test_year == start_year+1:
            torch.save(model.state_dict(), f'./model_weights/original.pth')

        mode = 'train'
        if mode == 'train':
            train(test_year)
        elif mode == 'test':
            test(test_year)
