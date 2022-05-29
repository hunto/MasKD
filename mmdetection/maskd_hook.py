from mmcv.runner.hooks import HOOKS, Hook

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.parallel import MMDistributedDataParallel

from maskd import MaskModule


@HOOKS.register_module()
class MasKDHook(Hook):
    def __init__(self, module_name='', channels=[], num_tokens=6, weight_mask=True):
        self.module_name = module_name
        self.channels = channels
        self.num_tokens = num_tokens
        self.weight_mask = weight_mask
        self.norms = []
        self.div_losses = []

    def before_run(self, runner):
        # init mask modules
        self.mask_modules = nn.ModuleList([MaskModule(c, self.num_tokens, self.weight_mask)
                                               for c in self.channels])
        self.mask_modules.cuda()
        assert isinstance(runner.model, MMDistributedDataParallel)
        runner.model.module.add_module('mask_modules', self.mask_modules)
        # fix the weights of model
        self._fix_weights(runner)
        runner.model = MMDistributedDataParallel(
                           runner.model.module,
                           device_ids=[torch.cuda.current_device()],
                           find_unused_parameters=True)
        # register module forward hook
        module = None
        for k, m in runner.model.module.named_modules():
            if k == self.module_name:
                module = m
                break
        assert module is not None, f'Cannot find module {self.module_name}'
        module.register_forward_hook(self._module_forward_hook)

    def after_train_iter(self, runner):
        # for verbose
        log_vars = {}
        for i, loss in enumerate(self.div_losses):
            log_vars[f'mask_div_loss_{i}'] = loss.item()
        runner.log_buffer.update(log_vars, runner.outputs['num_samples'])
        # add divergence loss to task loss
        runner.outputs['loss'] += sum(self.div_losses)
        self.div_losses = []

    def before_train_iter(self, runner):
        self.div_losses = []
        for m in self.norms:
            m.eval()

    def _module_forward_hook(self, module, input, output):
        assert len(output) == len(self.channels)
        masked_outputs = []
        for mask_module, out in zip(self.mask_modules, output):
            masked_out, div_loss = mask_module(out)
            masked_outputs.append(masked_out)
            self.div_losses.append(div_loss)
        return tuple(masked_outputs)

    def _fix_weights(self, runner):
        # ignore grads
        for n, p in runner.model.module.named_parameters():
            p.requires_grad = False

        opt_params = set()
        for k, m in runner.model.module.named_modules():
            if isinstance(m, MaskModule):
                runner.logger.info(f'Mask module: {k}')
                for p in m.parameters():
                    p.requires_grad = True
                    opt_params.add(p)
            if isinstance(m, _BatchNorm):
                self.norms.append(m)

        # add mask modules into optimizer
        runner.optimizer.param_groups = [runner.optimizer.param_groups[0]]
        runner.optimizer.param_groups[0]['params'] = list(self.mask_modules.parameters())