
import math
import torch
import torch.nn.functional as F
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

EPS = 1e-6

class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 search_params=None):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.search_params = search_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        env_params = dict(self.env_params)
        env_params.setdefault('device', device)
        self.env = Env(**env_params)
        self.local_search = None
        if self.trainer_params.get('local_search'):
            from TSPLocalSearch import TSPLocalSearch as LocalSearch
            self.local_search = LocalSearch(**self.search_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Loss Components
        self.loss_type = trainer_params['loss_type']
        self.alpha = trainer_params['alpha']
        sparse_bt_defaults = {
            'enable': True,
            'best_anchor_count': 6,
            'adjacent_top_count': 32,
            'global_pair_num': 32,
        }
        self.sparse_bt_params = sparse_bt_defaults
        self.sparse_bt_params.update(trainer_params.get('sparse_bt', {}))
        self.rl_mix_params = {
            'enable': True,
            'weight': 0.2,
        }
        self.rl_mix_params.update(trainer_params.get('rl_mix', {}))
        self.pair_schedule = trainer_params.get('pair_schedule', [])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        self._sparse_pair_cache = {}
        self._last_active_schedule_signature = None

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()
            active_train_params = self._get_active_train_params(epoch)
            self._log_active_train_params(active_train_params)

            # Train
            train_score, train_loss, pairwise_accuracy, avg_logprob_avg = self._train_one_epoch(epoch, active_train_params)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('pairwise_accuracy', epoch, pairwise_accuracy)
            self.result_log.append('avg_logprob_avg', epoch, avg_logprob_avg)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['pairwise_accuracy'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['avg_logprob_avg'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['pairwise_accuracy'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['avg_logprob_avg'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, active_train_params):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        pairwise_acc_AM = AverageMeter()
        avg_logprob_avg_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        step = 0
        while episode < train_num_episode:
            step = step + 1
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss, pairwise_acc, avg_logprob_avg = self._train_one_batch(batch_size, step, active_train_params)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            pairwise_acc_AM.update(pairwise_acc, batch_size)
            avg_logprob_avg_AM.update(avg_logprob_avg, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f},'
                                     '  PairAcc: {:.4f},  AvgLogProb: {:.4f}'
                                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                                score_AM.avg, loss_AM.avg, pairwise_acc_AM.avg, avg_logprob_avg_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f},'
                         '  PairAcc: {:.4f},  AvgLogProb: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, pairwise_acc_AM.avg, avg_logprob_avg_AM.avg))

        return score_AM.avg, loss_AM.avg, pairwise_acc_AM.avg, avg_logprob_avg_AM.avg

    def _train_one_batch(self, batch_size, step, active_train_params):
        # Augmentation
        ###############################################
        if self.trainer_params['augmentation_enable']:
            aug_factor = self.trainer_params['aug_factor']
        else:
            aug_factor = 1

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size // aug_factor, aug_factor=aug_factor)

        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        route_log_prob = torch.zeros(size=(batch_size, self.env.pomo_size), device=self.device)
        # shape: (batch, pomo)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            route_log_prob = route_log_prob + prob.clamp_min(EPS).log()

        pairwise_acc, avg_logprob_avg = self._compute_pairwise_diagnostics(
            reward,
            route_log_prob,
            active_train_params['sparse_bt'],
        )

        # Loss
        ###############################################
        if self.loss_type == 'po_ls_loss':
            if self.local_search is None:
                raise RuntimeError('local_search is enabled but TSPLocalSearch was not initialized.')
            loss = self.preference_among_two_opt_loss_fn(
                reward,
                route_log_prob=route_log_prob,
                sparse_bt_params=active_train_params['sparse_bt'],
                alpha=active_train_params['alpha'],
            )
        elif self.loss_type == 'po_loss':
            loss = self.preference_among_pomo_loss_fn(
                reward,
                route_log_prob=route_log_prob,
                sparse_bt_params=active_train_params['sparse_bt'],
                alpha=active_train_params['alpha'],
            )
        elif self.loss_type == 'pl_loss':
            loss = self.rank_among_pomo_loss_fn(reward, route_log_prob=route_log_prob)
        elif self.loss_type == 'rl_loss':
            loss = self.rl_loss_fn(reward, route_log_prob=route_log_prob)
        else:
            raise NotImplementedError

        log_loss = loss
        rl_weight = active_train_params['rl_weight']
        if self.loss_type != 'rl_loss' and rl_weight > 0:
            loss = loss + rl_weight * self.rl_loss_fn(reward, route_log_prob=route_log_prob)

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        loss = loss / self.trainer_params['optimizer_step_interval']
        loss.backward()
        if step % self.trainer_params['optimizer_step_interval'] == 0:
            self.optimizer.step()
            self.model.zero_grad()
        return score_mean.item(), log_loss.item(), pairwise_acc, avg_logprob_avg

    def _compute_pairwise_diagnostics(self, reward, route_log_prob, sparse_bt_params):
        if sparse_bt_params['enable']:
            reward_sorted, route_log_prob_sorted = self._sort_reward_and_log_prob(reward, route_log_prob)
            pair_rows, pair_cols = self._get_sparse_pair_indices(reward.size(1), reward.device, sparse_bt_params)
            valid_mask = reward_sorted[:, pair_rows] > reward_sorted[:, pair_cols]

            if valid_mask.any():
                pairwise_correct = (route_log_prob_sorted[:, pair_rows] > route_log_prob_sorted[:, pair_cols]) == valid_mask
                pairwise_acc = pairwise_correct[valid_mask].float().mean().item()
            else:
                pairwise_acc = 0.0
        else:
            reward_diff = reward[:, :, None] - reward[:, None, :]
            valid_mask = reward_diff != 0

            if valid_mask.any():
                log_prob_diff = route_log_prob[:, :, None] - route_log_prob[:, None, :]
                pairwise_correct = (log_prob_diff > 0) == (reward_diff > 0)
                pairwise_acc = pairwise_correct[valid_mask].float().mean().item()
            else:
                pairwise_acc = 0.0

        avg_logprob_avg = route_log_prob.mean().item()
        return pairwise_acc, avg_logprob_avg

    def _sort_reward_and_log_prob(self, reward, route_log_prob):
        sorted_index = torch.sort(reward, dim=1, descending=True)[1]
        reward_sorted = reward.gather(1, sorted_index)
        route_log_prob_sorted = route_log_prob.gather(1, sorted_index)
        return reward_sorted, route_log_prob_sorted

    def _get_active_train_params(self, epoch):
        active_sparse_bt = dict(self.sparse_bt_params)
        active_rl_weight = self.rl_mix_params['weight'] if self.rl_mix_params.get('enable', True) else 0.0

        for schedule_item in self.pair_schedule:
            start_epoch = schedule_item.get('start_epoch', 1)
            end_epoch = schedule_item.get('end_epoch', self.trainer_params['epochs'])
            if start_epoch <= epoch <= end_epoch:
                for key in ('enable', 'best_anchor_count', 'adjacent_top_count', 'global_pair_num'):
                    if key in schedule_item:
                        active_sparse_bt[key] = schedule_item[key]
                if 'rl_weight' in schedule_item:
                    active_rl_weight = schedule_item['rl_weight']
                if 'alpha' in schedule_item:
                    active_alpha = schedule_item['alpha']
                else:
                    active_alpha = self.alpha
                break
        else:
            active_alpha = self.alpha

        return {
            'sparse_bt': active_sparse_bt,
            'rl_weight': active_rl_weight,
            'alpha': active_alpha,
        }

    def _log_active_train_params(self, active_train_params):
        sparse_bt_params = active_train_params['sparse_bt']
        signature = (
            sparse_bt_params.get('enable', False),
            sparse_bt_params.get('best_anchor_count', 0),
            sparse_bt_params.get('adjacent_top_count', 0),
            sparse_bt_params.get('global_pair_num', 0),
            round(active_train_params['rl_weight'], 6),
            round(active_train_params['alpha'], 6),
        )
        if signature != self._last_active_schedule_signature:
            self.logger.info(
                'Active train params: sparse_bt=%s, best_anchor=%d, adjacent_top=%d, global_pair_num=%d, rl_weight=%.4f, alpha=%.4f',
                sparse_bt_params.get('enable', False),
                sparse_bt_params.get('best_anchor_count', 0),
                sparse_bt_params.get('adjacent_top_count', 0),
                sparse_bt_params.get('global_pair_num', 0),
                active_train_params['rl_weight'],
                active_train_params['alpha'],
            )
            self._last_active_schedule_signature = signature

    def _get_sparse_pair_indices(self, pomo_size, device, sparse_bt_params):
        key = (
            pomo_size,
            device.type,
            device.index,
            sparse_bt_params.get('enable', False),
            sparse_bt_params.get('best_anchor_count', 0),
            sparse_bt_params.get('adjacent_top_count', 0),
            sparse_bt_params.get('global_pair_num', 0),
        )
        cached = self._sparse_pair_cache.get(key)
        if cached is not None:
            return cached

        pair_mask = torch.zeros((pomo_size, pomo_size), dtype=torch.bool, device=device)

        if pomo_size > 1:
            best_anchor_count = min(sparse_bt_params.get('best_anchor_count', 0), pomo_size - 1)
            if best_anchor_count > 0:
                pair_mask[0, 1:best_anchor_count + 1] = True

            adjacent_top_count = min(sparse_bt_params.get('adjacent_top_count', 0), pomo_size)
            if adjacent_top_count > 1:
                pair_rows = torch.arange(adjacent_top_count - 1, device=device)
                pair_mask[pair_rows, pair_rows + 1] = True

            global_pair_num = sparse_bt_params.get('global_pair_num', 0)
            if global_pair_num > 0:
                self._add_global_stratified_pairs(pair_mask, global_pair_num, device)

        pair_indices = pair_mask.nonzero(as_tuple=True)
        self._sparse_pair_cache[key] = pair_indices
        return pair_indices

    def _add_global_stratified_pairs(self, pair_mask, global_pair_num, device):
        pomo_size = pair_mask.size(0)
        top_end = max(2, int(round(pomo_size * 0.2)))
        mid_end = max(top_end + 1, int(round(pomo_size * 0.6)))

        top_pool = torch.arange(0, top_end, device=device)
        mid_pool = torch.arange(top_end, mid_end, device=device)
        bottom_pool = torch.arange(mid_end, pomo_size, device=device)

        segment_specs = [
            (top_pool, mid_pool, int(math.ceil(global_pair_num * 0.4))),
            (top_pool, bottom_pool, int(math.ceil(global_pair_num * 0.4))),
            (mid_pool, bottom_pool, max(0, global_pair_num - 2 * int(math.ceil(global_pair_num * 0.4)))),
        ]

        for left_pool, right_pool, requested_count in segment_specs:
            if requested_count <= 0 or left_pool.numel() == 0 or right_pool.numel() == 0:
                continue
            count = min(requested_count, left_pool.numel(), right_pool.numel())
            left_selected = self._select_evenly_spaced(left_pool, count)
            right_selected = self._select_evenly_spaced(right_pool, count)
            count = min(left_selected.numel(), right_selected.numel())
            if count > 0:
                pair_mask[left_selected[:count], right_selected[:count]] = True

    def _select_evenly_spaced(self, index_pool, count):
        if count <= 0 or index_pool.numel() == 0:
            return index_pool[:0]
        if count >= index_pool.numel():
            return index_pool
        positions = torch.linspace(0, index_pool.numel() - 1, steps=count, device=index_pool.device)
        indices = positions.round().long()
        return index_pool[indices]

    def rl_loss_fn(self, reward, prob_list=None, route_log_prob=None):
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        if route_log_prob is None:
            if prob_list is None:
                raise ValueError('Either prob_list or route_log_prob must be provided.')
            log_prob = prob_list.clamp_min(EPS).log().sum(dim=2)
        else:
            log_prob = route_log_prob
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        return loss_mean
    
    def kl_loss_fn(self, prob, reference_prob):
        log_prob = torch.log(prob)
        reference_log_prob = torch.log(reference_prob)
        return torch.mean(prob * (log_prob - reference_log_prob))
    
    def preference_among_pomo_loss_fn(self, reward, prob=None, route_log_prob=None, sparse_bt_params=None, alpha=None): # BT
        if route_log_prob is None:
            if prob is None:
                raise ValueError('Either prob or route_log_prob must be provided.')
            route_log_prob = torch.log(prob.clamp_min(EPS)).sum(2)

        if sparse_bt_params is None:
            sparse_bt_params = self.sparse_bt_params
        if alpha is None:
            alpha = self.alpha

        if sparse_bt_params['enable']:
            reward_sorted, route_log_prob_sorted = self._sort_reward_and_log_prob(reward, route_log_prob)
            pair_rows, pair_cols = self._get_sparse_pair_indices(reward.size(1), reward.device, sparse_bt_params)
            valid_mask = reward_sorted[:, pair_rows] > reward_sorted[:, pair_cols]

            if not valid_mask.any():
                return route_log_prob.sum() * 0

            pair_logits = route_log_prob_sorted[:, pair_rows] - route_log_prob_sorted[:, pair_cols]
            return -F.logsigmoid(alpha * pair_logits[valid_mask]).mean()

        preference = reward[:, :, None] > reward[:, None, :]
        log_prob_pair = route_log_prob[:, :, None] - route_log_prob[:, None, :]
        pf_log = torch.log(F.sigmoid(alpha * log_prob_pair))
        loss = -torch.mean(pf_log * preference)
        return loss
    
    def rank_among_pomo_loss_fn(self, reward, prob=None, route_log_prob=None): # PL
        sorted_index = torch.sort(reward, dim=1, descending=True)[1]
        # shape: (batch, pomo)
        if route_log_prob is None:
            if prob is None:
                raise ValueError('Either prob or route_log_prob must be provided.')
            route_log_prob = torch.log(prob.clamp_min(EPS)).sum(2)
        log_prob = self.alpha * route_log_prob
        max_log_prob = log_prob.max(1, keepdim=True)[0]
        log_prob = log_prob - max_log_prob
        exp_log_prob = torch.exp(log_prob)
        one_hot = F.one_hot(sorted_index).to(torch.float)
        # shape: (batch, pomo, pomo)
        till_mat = torch.tril(torch.ones_like(one_hot))
        sum_exp = (till_mat @ one_hot @ exp_log_prob[:, :, None]).squeeze(-1)
        # shape: (batch, pomo)
        loss = torch.mean(torch.log(exp_log_prob) - torch.log(sum_exp))

        return loss

    def preference_among_two_opt_loss_fn(self, reward, probs=None, route_log_prob=None, sparse_bt_params=None, alpha=None):
        dist = self.env.get_distmat()
        route_info = self.local_search.search(self.env.selected_node_list, reward, dist, self.env.problems)
        
        search_probs = self.model.route_forward(route_info) + EPS
        if route_log_prob is None:
            route_log_prob = torch.log(probs.clamp_min(EPS)).sum(2)
        search_log_prob = torch.log(search_probs.clamp_min(EPS)).sum(2)
        route_log_prob = torch.cat((route_log_prob, search_log_prob), dim=1)
        reward = torch.cat((reward, route_info.reward), dim=1)
        return self.preference_among_pomo_loss_fn(
            reward,
            route_log_prob=route_log_prob,
            sparse_bt_params=sparse_bt_params,
            alpha=alpha,
        )
