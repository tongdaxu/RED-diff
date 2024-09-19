import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from compressai.models import ScaleHyperprior
from compressai.zoo import load_state_dict
import torch.nn.functional as F
import argparse
from compressai.models import ScaleHyperprior
from compressai.entropy_models import EntropyBottleneck, GaussianConditional, EntropyModel
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from algos import build_algo
from datasets import build_loader
from models import build_model
from hydra.core.hydra_config import HydraConfig
from utils.distributed import get_logger, init_processes, common_init
from models.diffusion import Diffusion
from models.classifier_guidance_model import ClassifierGuidanceModel
from utils.functions import get_timesteps
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Quantizator_SGA(nn.Module):
    """
    https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/c9b5c1354a38e0bb505fc34c6c8f27170f62a75b/sga.py#L110
    Stochastic Gumbeling Annealing
    sample() has no grad, so we choose STE to backward. We can also try other estimate func.
    """

    def __init__(self, gap=1000, c=0.002):
        super(Quantizator_SGA, self).__init__()
        self.gap = gap
        self.c = c

    def annealed_temperature(self, t, r, ub, lb=1e-8, backend=np, scheme='exp', **kwargs):
        """
        Return the temperature at time step t, based on a chosen annealing schedule.
        :param t: step/iteration number
        :param r: decay strength
        :param ub: maximum/init temperature
        :param lb: small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
        :param backend: np or tf
        :param scheme:
        :param kwargs:
        :return:
        """
        default_t0 = kwargs.get('t0')

        if scheme == 'exp':
            tau = backend.exp(-r * t)
        elif scheme == 'exp0':
            # Modified version of above that fixes temperature at ub for initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = ub * backend.exp(-r * (t - t0))
        elif scheme == 'linear':
            # Cool temperature linearly from ub after the initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = -r * (t - t0) + ub
        else:
            raise NotImplementedError

        if backend is None:
            return min(max(tau, lb), ub)
        else:
            return backend.minimum(backend.maximum(tau, lb), ub)

    def forward(self, input, it=None, mode=None, total_it=None):
        if mode == "training":
            assert it is not None
            x_floor = torch.floor(input)
            x_ceil = torch.ceil(input)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)

            eps = 1e-5

            annealing_scheme = 'exp0'
            annealing_rate = 1e-3  # default annealing_rate = 1e-3
            t0 = int(total_it * 0.35)  # default t0 = 700 for 2000 iters
            T_ub = 0.5

            T = self.annealed_temperature(it, r=annealing_rate, ub=T_ub, scheme=annealing_scheme, t0=t0)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

            rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            rx = F.softmax(rx_logits, dim=-1)  # just for observation in tensorboard
            rx_dist = torch.distributions.RelaxedOneHotCategorical(T, rx)

            rx_sample = rx_dist.rsample()

            x_tilde = torch.sum(x_bds * rx_sample, dim=-1)
            return x_tilde
        else:
            return torch.round(input)

class EntropyBottleneckNoQuant(EntropyBottleneck):
    def __init__(self, channels):
        super().__init__(channels)
        self.sga = Quantizator_SGA()

    def forward(self, x_quant):
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)
        likelihood = self._likelihood(x_quant)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood


class GaussianConditionalNoQuant(GaussianConditional):
    def __init__(self, scale_table):
        super().__init__(scale_table=scale_table)

    def forward(self, x_quant, scales, means):
        likelihood = self._likelihood(x_quant, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return likelihood


class ScaleHyperpriorSGA(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)
        self.entropy_bottleneck = EntropyBottleneckNoQuant(N)
        self.gaussian_conditional  = GaussianConditionalNoQuant(None)
        self.sga = Quantizator_SGA()

    def quantize(self, inputs, mode, means=None, it=None, tot_it=None):
        if means is not None:
            inputs = inputs - means
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            outputs = inputs + noise
        elif mode == "round":
            outputs = torch.round(inputs)
        elif mode == "sga":
            outputs = self.sga(inputs, it, "training", tot_it)
        else:
            assert(0)
        if means is not None:
            outputs = outputs + means
        return outputs

    def forward(self, x, mode, y_in=None, z_in=None, it=None, tot_it=None):
        if mode == "init":
            y = self.g_a(x)
            z = self.h_a(torch.abs(y))
        else:
            y = y_in
            z = z_in
        if mode == "init" or mode == "round":
            y_hat = self.quantize(y, "round")
            z_hat = self.quantize(z, "round")
        elif mode == "noise":
            y_hat = self.quantize(y, "noise")
            z_hat = self.quantize(z, "noise")
        elif mode =="sga":
            y_hat = self.quantize(y, "sga", None, it, tot_it)
            z_hat = self.quantize(z, "sga", None, it, tot_it)
        else:
            assert(0)
        z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_likelihoods = self.gaussian_conditional(y_hat, scales_hat, None)
        x_hat = self.g_s(y_hat)
        return {
            "y": y.detach().clone(),
            "z": z.detach().clone(), 
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class KodakDataset(torch.utils.data.Dataset):
    def __init__(self, kodak_root):
        self.img_dir = kodak_root
        self.img_fname = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.img_fname)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_fname[idx])
        image = torchvision.io.read_image(img_path)
        image = image.to(dtype=torch.float32) / 255.0
        return image * 2.0 - 1.0


class SuperResolutionOperator(nn.Module):
    def __init__(self, scale_factor):
        super(SuperResolutionOperator, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        y = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bicubic', antialias=True)
        return y

def psnr(mse):
    return 10*torch.log10((255**2) / mse)

def main(cfg: DictConfig):

    lams = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483,0.0932,0.1800]
    q = int(0)
    Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
    N, M = Ns[q], Ms[q]

    op = SuperResolutionOperator(4).cuda()
    
    model_path = "./bmshj2018-hyperprior-1-7eb97409.pth.tar"
    model = ScaleHyperpriorSGA(N, M)
    model_dict = load_state_dict(torch.load(model_path))
    model.load_state_dict(model_dict)
    model = model.cuda()

    dataset = KodakDataset(kodak_root="/NEW_EDS/JJ_Group/xutd/common_datasets/imagenet_256x256/val1k")
    dataloader = torch.utils.data.DataLoader(dataset)

    model.eval()
    bpp_init_avg, mse_init_avg, psnr_init_avg, rd_init_avg = 0, 0, 0, 0
    bpp_post_avg, mse_post_avg, psnr_post_avg, rd_post_avg = 0, 0, 0, 0

    tot_it = 1000
    lr = 1e-1

    diffmodel, classifier = build_model(cfg)
    diffmodel.eval()
    if classifier is not None:
        classifier.eval()

    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ClassifierGuidanceModel(diffmodel, classifier, diffusion, cfg)
    algo = build_algo(cg_model, cfg)
    for param in diffmodel.parameters():
        param.requires_grad = False

    ts = get_timesteps(cfg)
    ss = [-1] + list(ts[:-1])

    ts = ts[::-1]
    ss = ss[::-1]

    for idx, img in enumerate(dataloader):

        img = img.cuda()
        img_h, img_w = img.shape[2], img.shape[3]
        img_pixnum = img_h * img_w

        mu = torch.randn_like(img)
        mu.requires_grad = True

        y = op(img)
        # first round
        opt = torch.optim.Adam([mu], lr=lr)

        for it in range(tot_it):

            opt.zero_grad()   
            
            # start diffusion loss
            n = 1
            ti = ts[it]
            si = ss[it]

            t = torch.ones(n).to(mu.device).long() * ti
            s = torch.ones(n).to(mu.device).long() * si
            alpha_t = algo.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = algo.diffusion.alpha(s).view(-1, 1, 1, 1)
            
            sigma_x0 = algo.sigma_x0  #0.0001
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)

            x0_pred = mu + sigma_x0*noise_x0
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
            
            #scale = 0.0
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * algo.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            #xt = xt.clone().to('cuda').requires_grad_(True)
            if algo.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            with torch.no_grad():
                et, x0_hat = algo.model(xt, None, t, scale=scale)   #et, x0_pred
                if not algo.awd:
                    et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
                et = et.detach()

            snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()  #1d torch tensor

            loss_noise = torch.mul((et - noise_xt).detach(), mu).mean()
            rdcost = loss_noise * snr_inv * 1.0 + F.mse_loss(op(mu), y)
            # rdcost = F.mse_loss(op(mu), y)
            rdcost.backward()
            opt.step()

            print("{0} rdcost: {1:.4f}".format(it, rdcost.item()))

        torchvision.utils.save_image(y, "input.png", normalize=True, value_range=(-1,1))
        torchvision.utils.save_image(img, "src.png", normalize=True, value_range=(-1,1))
        torchvision.utils.save_image(mu, "recon.png", normalize=True, value_range=(-1,1))
        torchvision.utils.save_image(op(mu), "re_input.png", normalize=True, value_range=(-1,1))
        assert(0)

def main_codec(cfg: DictConfig):

    lams = [0.0018,0.0035,0.0067,0.0130,0.0250,0.0483,0.0932,0.1800]
    q = int(0)
    Ns, Ms = [128,128,128,128,128,192,192,192], [192,192,192,192,192,320,320,320]
    N, M = Ns[q], Ms[q]

    op = SuperResolutionOperator(4).cuda()
    
    model_path = "./bmshj2018-hyperprior-1-7eb97409.pth.tar"
    model = ScaleHyperpriorSGA(N, M)
    model_dict = load_state_dict(torch.load(model_path))
    model.load_state_dict(model_dict)
    model = model.cuda()

    dataset = KodakDataset(kodak_root="/NEW_EDS/JJ_Group/xutd/common_datasets/imagenet_256x256/val1k")
    dataloader = torch.utils.data.DataLoader(dataset)

    model.eval()
    bpp_init_avg, mse_init_avg, psnr_init_avg, rd_init_avg = 0, 0, 0, 0
    bpp_post_avg, mse_post_avg, psnr_post_avg, rd_post_avg = 0, 0, 0, 0

    tot_it = 100
    lr = 5e-3

    diffmodel, classifier = build_model(cfg)
    diffmodel.eval()
    if classifier is not None:
        classifier.eval()

    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ClassifierGuidanceModel(diffmodel, classifier, diffusion, cfg)
    algo = build_algo(cg_model, cfg)
    for param in diffmodel.parameters():
        param.requires_grad = False

    ts = get_timesteps(cfg)
    ss = [-1] + list(ts[:-1])

    ts = ts[::-1]
    ss = ss[::-1]

    for idx, img in enumerate(dataloader):

        img = img.cuda()
        img_h, img_w = img.shape[2], img.shape[3]
        img_pixnum = img_h * img_w

        with torch.no_grad():
            ret_dict = model((img + 1.0) / 2.0, "init")
        bpp_init = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        mse_init = F.mse_loss((img + 1.0) / 2.0, ret_dict["x_hat"]) * (255 ** 2)
        rd_init = bpp_init + lams[q] * mse_init

        torchvision.utils.save_image(ret_dict["x_hat"] * 2.0 - 1.0, "init.png", normalize=True, value_range=(-1,1))

        y, z = nn.parameter.Parameter(ret_dict["y"]), nn.parameter.Parameter(ret_dict["z"])
        opt = torch.optim.Adam([y], lr=lr)
        print("init: bpp: {}, mse: {}".format(bpp_init.item(), mse_init.item()))

        for it in tqdm(range(tot_it)):

            opt.zero_grad()   

            ret_dict = model((img + 1.0) / 2.0, "sga", y, z, it, tot_it)
            bpp = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) + \
                  torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
            mse = F.mse_loss((img + 1.0) / 2.0, ret_dict["x_hat"]) * (255 ** 2)
            mu = ret_dict["x_hat"] * 2.0 - 1.0
            # start diffusion loss
            n = 1
            ti = ts[it]
            si = ss[it]

            t = torch.ones(n).to(mu.device).long() * ti
            s = torch.ones(n).to(mu.device).long() * si
            alpha_t = algo.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = algo.diffusion.alpha(s).view(-1, 1, 1, 1)
            
            sigma_x0 = algo.sigma_x0  #0.0001
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)

            x0_pred = mu + sigma_x0*noise_x0
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_xt
            
            #scale = 0.0
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * algo.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            #xt = xt.clone().to('cuda').requires_grad_(True)
            if algo.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            with torch.no_grad():
                et, x0_hat = algo.model(xt, None, t, scale=scale)   #et, x0_pred
                if not algo.awd:
                    et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
                et = et.detach()

            snr_inv = (1-alpha_t[0]).sqrt()/alpha_t[0].sqrt()  #1d torch tensor

            loss_noise = torch.mul((et - noise_xt).detach(), mu).mean()
            rdcost = loss_noise * snr_inv * 10.0 + bpp + lams[q] * mse
            # rdcost = bpp + lams[q] * mse
            rdcost.backward()
            opt.step()

            # print("{0} rdcost: {1:.4f}".format(it, rdcost.item()))
            # print("{}: bpp: {}, mse: {}".format(it, bpp.item(), mse.item()))

        with torch.no_grad():
            ret_dict = model(img, "round", y, z)

        bpp_post = torch.sum(-torch.log2(ret_dict["likelihoods"]["y"])) / (img_pixnum) +\
                   torch.sum(-torch.log2(ret_dict["likelihoods"]["z"])) / (img_pixnum)
        mse_post = F.mse_loss((img + 1.0) / 2.0, ret_dict["x_hat"]) * (255 ** 2)
        print("post bpp: {}, mse: {}".format(bpp_post.item(), mse_post.item()))

        torchvision.utils.save_image(img, "src.png", normalize=True, value_range=(-1,1))
        torchvision.utils.save_image(ret_dict["x_hat"] * 2.0 - 1.0, "recon.png", normalize=True, value_range=(-1,1))
        assert(0)

@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp")
def main_dist(cfg: DictConfig):
    size = 1
    cwd = HydraConfig.get().runtime.output_dir
    init_processes(0, size, main_codec, cfg, cwd)

if __name__ == "__main__":
    main_dist()
