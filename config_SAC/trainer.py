import torch
import random
import numpy as np
import gym

from baseline.baseline_trainer import Policy
from config_SAC.Agent import SAC_Agent
from baseline.utils import get_optimizer,get_optimizer_weight,calculate_global_norm


class action_space:

    def __init__(self, action_size):
        self.shape = [action_size]
        self.low = -1
        self.high = 1

class SAC_Trainer(Policy):

    def __init__(self, file_name):
        super(SAC_Trainer, self).__init__(file_name)
        if 'fixed_temperature' in self.parm.keys():
            self.fixed_temperature = self.parm['fixed_temperature'] == 'True'
            if 'temperature_parameter' in self.parm.keys():
                self.temperature_value = self.parm['temperature_parameter']
        else:
            self.fixed_temperature = False
        torch.manual_seed(self.parm['seed'])
        np.random.seed(self.parm['seed'])

        self.agent = SAC_Agent(self.agent_parms).train()
        if self.parm['load_path'] != "None":
            self.agent.load_state_dict(torch.load(self.parm['load_path'], map_location=self.parm['gpu_name']))
        self.target_agent = SAC_Agent(self.agent_parms)

        self.actor_feature , self.actor, self.policy, self.critic1, self.critic2, self.temperature = self.agent.actor_Feature,self.agent.actor,self.agent.policy,self.agent.critic, self.agent.critic2, self.agent.Temperature

        self.target_critic1, self.target_critic2 = self.target_agent.critic, self.target_agent.critic2

        self.target_critic1.load_state_dict(self.critic1.state_dict(),strict=False)
        self.target_critic2.load_state_dict(self.critic2.state_dict(),strict=False)
        self.actor_optimizer, self.critic_optimizer1, self.critic_optimizer2, self.temperature_optimizer = self.generate_optimizer()

        self.eval_env = gym.make(self.parm['env_name'])
        self.eval_obs_set = []

        self.flatten_state_size = [-1,self.state_size[0] * self.state_size[1]]

        self.best = 3000
        self.gradient_steps = self.parm['gradient_steps']


    def generate_optimizer(self):

        weights = list(self.actor.parameters())+list(self.policy.parameters())
        weights = weights + list(self.actor_feature.parameters())
        actor_optimizer = get_optimizer_weight(self.optimizer_parms['actor'], weights)
        critic_optimizer1 = get_optimizer(self.optimizer_parms['critic'], self.critic1)
        critic_optimizer2 = get_optimizer(self.optimizer_parms['critic'], self.critic2)
        if self.fixed_temperature == False:
            temperature_optimizer = get_optimizer_weight(self.optimizer_parms['temperature'], [self.temperature])
        else:
            temperature_optimizer = None

        return actor_optimizer,critic_optimizer1, critic_optimizer2, temperature_optimizer

    def get_action(self, state):
        if ~torch.is_tensor(state):
            state = torch.tensor(state).to(self.gpu).float()

        size = [-1]
        for i in self.state_size:
            size.append(i)
        state = state.view(size)
        action,log_std, critics,_ = self.agent.eval().forward(state)

        return action.to(self.cpu).detach().numpy()

    def get_action_deterministic(self,state):
        if ~torch.is_tensor(state):
            state = torch.tensor(state).to(self.gpu).float()

        size = [-1, self.state_size[0]*self.state_size[1]]
        state = state.view(size)
        actor_feature = self.agent.actor_Feature(state)
        mean = torch.tanh(self.actor(actor_feature))

        return mean.to(self.cpu).detach().numpy()

    def target_network_update(self,step):
        if self.c_mode:
            with torch.no_grad():
                for t_pa1,t_pa2, pa1,pa2 in zip(self.target_critic1.parameters(),self.target_critic2.parameters(), self.critic1.parameters(),self.critic2.parameters()):

                    temp1 = self.tau * pa1 + (1 - self.tau) * t_pa1
                    temp2 = self.tau * pa2 + (1 - self.tau) * t_pa2
                    t_pa1.copy_(temp1)
                    t_pa2.copy_(temp2)

    def preprocess_state_eval(self, obs):

        state = np.zeros(self.parm['state_size'])

        self.obs_set.append(obs)

        for i in range(self.parm['state_size'][0]):
            state[i] = self.obs_set[i]

        if self.agent.mode == 'Image':
            state = np.uint8(state)
        return state

    def zero_grad(self):
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        self.actor_optimizer.zero_grad()
        if self.fixed_temperature == False:
            self.temperature_optimizer.zero_grad()

    def train(self):
        mini_batch = random.sample(self.replay_memory, self.batch_size)
        self.agent = self.agent.train()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append([mini_batch[i][1]])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])
        actions = torch.tensor(actions).view(self.batch_size,self.action_size).to(self.gpu).float()

        next_states = torch.tensor(next_states).view(self.flatten_state_size).float().to(self.gpu)
        states = torch.tensor(states).view(self.flatten_state_size).float().to(self.gpu)
        next_actions,log_prob,_,entropy = self.agent.eval().forward(next_states)
        next_state_action = torch.cat((next_states,next_actions),dim=1)

        target1 = self.target_critic1.eval()(next_state_action)
        target2 = self.target_critic2.eval()(next_state_action)

        temperature = self.temperature.detach().view((-1,1))
        if self.fixed_temperature:
            alpha = self.temperature_value
        else:
            alpha = temperature.exp()

        for i in range(self.batch_size):
            if dones[i]:
                target1[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
                target2[i] = torch.tensor(rewards[i]).to(self.gpu).detach().float()
            else:
                temp = -alpha * log_prob[i]
                target1[i] = rewards[i] + self.discount_factor * (target1[i]+temp)
                target2[i] = rewards[i] + self.discount_factor * (target2[i]+temp)

        if self.fixed_temperature:
            loss_critic1,loss_critic2, loss_policy, loss_temp = self.agent.loss(states.detach(), (target1.detach(),target2.detach()),actions.detach(),alpha=alpha)

        else:
            loss_critic1, loss_critic2, loss_policy, loss_temp = self.agent.loss(states.detach(),
                                                                                 (target1.detach_(), target2.detach()),
                                                                                 actions.detach())

        self.zero_grad()
        loss_policy.backward()

        self.critic1.zero_grad()
        self.critic2.zero_grad()

        loss_critic1.backward()
        loss_critic2.backward()
        if self.fixed_temperature:
            pass
        else:
            loss_temp.backward()
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        self.actor_optimizer.step()
        if self.fixed_temperature==False:
            self.temperature_optimizer.step()

        z = calculate_global_norm(self.actor_feature)
        z += calculate_global_norm(self.critic2)
        z += calculate_global_norm(self.critic1)
        z += calculate_global_norm(self.policy)
        z += calculate_global_norm(self.actor)




        return loss_policy,(loss_critic1+loss_critic2)/2,loss_temp, z,entropy.mean().detach().to(self.cpu).numpy()

    def eval(self,step):
        episode_reward = []
        for i in range(10):

            ob_eval = self.eval_env.reset()
            self.eval_obs_set = []
            state_ob = self.preprocess_state(ob_eval)
            a_eval = self.get_action_deterministic(state_ob)
            done = False

            _reward = 0

            while done == False:
                obs_eval, reward_eval, done, ___ = self.eval_env.step(a_eval)

                next_state_eval = self.preprocess_state_eval(obs_eval)
                n_a_eval = self.get_action_deterministic(next_state_eval)
                a_eval = n_a_eval
                _reward += reward_eval
                if done:
                    episode_reward.append(_reward)

        self.writer.add_scalar('Reward', np.array(episode_reward).mean(), step)
        if(np.array(episode_reward).mean()> self.best):
            self.best = np.array(episode_reward).mean()
            torch.save(self.agent.state_dict(), './save/01_SAC/best_humanoid_{}.pth'.format(str(self.best)))


    def run(self):

        losses = []
        rewards = []
        norms = []
        step = 0
        episode = 0

        breakvalue = 1
        while breakvalue:

            episode_loss_p = []
            episode_loss_c = []
            episode_loss_t = []
            episode_norm = []
            episode_entropy = []
            episode_reward = 0

            self.reset()


            ob = self.env.reset()
            state = self.preprocess_state(ob)
            done = False
            a = self.get_action(state)

            while done == False:


                obs, reward, done, _ = self.env.step(a)
                step += 1

                if (step+1)%1000 == 0:
                    self.eval(step)

                next_state = self.preprocess_state(obs)
                n_a = self.get_action(next_state)
                self.append_memory((state, a, reward*self.reward_scaling, next_state, done,n_a))
                if step >= self.start_step and self.inference_mode == False:
                    if step & self.parm['learning_freq'] == 0:
                        for zzz in range(self.gradient_steps):
                            loss_p,loss_c, loss_t, norm,entropy= self.train()
                            episode_loss_p.append(loss_p.to(self.cpu).detach().numpy())
                            episode_loss_c.append(loss_c.to(self.cpu).detach().numpy())
                            episode_loss_t.append(loss_t.to(self.cpu).detach().numpy())
                            episode_norm.append(norm)
                            episode_entropy.append(entropy)

                state = next_state
                a= n_a

                episode_reward += reward
                if self.parm['render_mode'] == 'True':
                    self.env.render()
                if step > self.start_step:
                    self.target_network_update(step)


                if done:
                    episode_loss_p = np.array(episode_loss_p).mean()
                    episode_loss_c = np.array(episode_loss_c).mean()
                    episode_loss_t = np.array(episode_loss_t).mean()
                    episode_norm = np.array(episode_norm).mean()
                    episode_entropy = np.array(episode_entropy).mean()

                    if self.fixed_temperature:
                        alpha = self.temperature_value
                    else:
                        alpha = self.agent.alpha.to(self.cpu).detach().numpy().mean()

                    self.writer.add_scalar('Policy_Loss', episode_loss_p, step)
                    self.writer.add_scalar('Critic_Loss', episode_loss_c, step)
                    self.writer.add_scalar('Temperature_Loss', episode_loss_t, step)

                    self.writer.add_scalar('Norm', episode_norm, step)
                    self.writer.add_scalar('Alpha', alpha,step)
                    self.writer.add_scalar('Entropy', episode_entropy,step)

                    losses.append(episode_loss_t + episode_loss_p + episode_loss_c)
                    rewards.append(episode_reward)
                    norms.append(episode_norm)
                    episode += 1

                    if (episode + 1) % self.show_episode == 0 or self.inference_mode:
                        losses = np.array(losses).mean()
                        rewards_ = np.array(rewards).mean()
                        norms = np.array(norms).mean()
                        if self.fixed_temperature:
                            alpha = self.temperature_value
                        else:
                            alpha =self.agent.alpha.to(self.cpu).detach().numpy().mean()
                        print(
                            'Episode : {:4d} // Step : {:5d} // Loss: {:3f} // Reward : {:3f} // Norm : {:3f} // alpha: {:3f}'.format(
                                episode + 1, step, losses, rewards_, norms,alpha))

                        losses = []
                        rewards = []
                        norms = []
                    if step > self.start_step and (episode + 1) % 10 == 0 and self.inference_mode == False:
                        torch.save(self.agent.state_dict(), self.save_path)