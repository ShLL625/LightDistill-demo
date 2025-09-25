import time
import numpy as np
import torch
import torch.nn as nn
import imageio.v3 as iio
import math
from torchvision.transforms import transforms

emap_res = [128, 256]
resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC) #resize from [128, 256] to [512, 1024] to match ALP resolution

#----------------------------------------------------------------------------
# image processing functions.
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def save_image(fn, x : np.ndarray):
    iio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8), compress_level=3)

def load_image(fn) -> np.ndarray:
    img = iio.imread(fn)
    return img.astype(np.float32) / 255
        
#----------------------------------------------------------------------------
# Distribution functions.
#----------------------------------------------------------------------------
    
def cart_to_equil(xyz, res):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    theta = torch.asin(torch.clamp(y, -1+1e-7, 1-1e-7))# * (y/abs(y))
    phi = torch.acos(torch.clamp(z / ((x**2 + z**2)**0.5), -1+1e-7, 1-1e-7)) * (x/abs(x))# * (((x>=0).float()-0.5)*2)
    
    gy = torch.round((1 - theta * 2 / np.pi) * (res[0] - 1) / 2)
    gx = torch.round((phi + np.pi/2) / np.pi * (res[1] - 1))
    
    equil_coord = torch.stack((gy, gx), dim=-1)
    
    return equil_coord

def equil_to_cart(res):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    reflvec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    
    return reflvec

def distribution(norm, wo, r, rgb):  
    bs = norm.shape[0]
    xyz = equil_to_cart([emap_res[0],emap_res[1]])
    xyz = xyz.reshape(xyz.shape[0]*xyz.shape[1], xyz.shape[2])
    norm = norm.repeat(1, xyz.shape[0]).reshape(xyz.shape[0]*bs, 3)
    wo = wo.repeat(1, xyz.shape[0]).reshape(xyz.shape[0]*bs, 3)
    r = r.repeat(1, xyz.shape[0]).reshape(xyz.shape[0]*bs, 1)
    rgb = rgb[..., 0:3].repeat(1, xyz.shape[0]).reshape(xyz.shape[0]*bs, 3)
    xyz = xyz.repeat(bs, 1)
    x, y, z = xyz[..., 0, None], xyz[..., 1, None], xyz[..., 2, None]
    wox, woy, woz = wo[..., 0, None], wo[..., 1, None], wo[..., 2, None]
    nx, ny, nz = norm[..., 0, None], norm[..., 1, None], norm[..., 2, None]
    
    dot = x*nx + y*ny + z*nz
    #NDF
    NoH = ((x+wox)*nx + (y+woy)*ny + (z+woz)*nz) / (((x+wox)**2 + (y+woy)**2 + (z+woz)**2)**0.5)
    D = (r**2/(np.pi*(NoH**2*(r**2-1.)+1.)**2))
    weight = torch.zeros(xyz.shape[0:-1], device='cuda')[..., None]
    threshald = 0.
    weight[(dot>threshald).reshape(dot.shape[0])] = dot[(dot>threshald).reshape(dot.shape[0])] * D[(dot>threshald).reshape(dot.shape[0])]
    
    rgb = (rgb * weight).reshape(bs, emap_res[0]*emap_res[1], 3)
    weight = weight.reshape(bs, emap_res[0]*emap_res[1], 1)
    rgb = torch.sum(rgb, 0)
    weight = torch.sum(weight, 0)
    
    return rgb, weight

#----------------------------------------------------------------------------
# LightDistill MLP.
#----------------------------------------------------------------------------

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons']), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons']), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims']), torch.nn.ReLU())
        self.net = torch.nn.Sequential(*net).cuda()
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        output = self.net(x.to(torch.float32))
        return output

class LightDistill(nn.Module):
    def __init__(self):
        super(LightDistill, self).__init__()
        
        self.B0 = torch.randn((64, 3), device = 'cuda')
        self.B1 = torch.randn((64, 3), device = 'cuda')
        self.B2 = torch.randn((64, 3), device = 'cuda')
        self.B3 = torch.randn((64, 3), device = 'cuda')
        self.B4 = torch.randn((64, 3), device = 'cuda')
        self.B5 = torch.randn((64, 1), device = 'cuda')
        mlp_cfg = {
            "n_input_dims" : 128,
            "n_output_dims" : 3,
            "n_hidden_layers" : 4,
            "n_neurons" : 128
        }
        self.mlp = _MLP(mlp_cfg)
        
    def PE_norm(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B0.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def PE_wo(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B1.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def PE_rgb(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B2.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def PE_kd(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B3.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def PE_ks(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B4.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def PE_r(self, xy):
        xy_proj = (2.*np.pi*xy) @ self.B5.T
        encoded_xy = torch.cat([torch.sin(xy_proj), torch.cos(xy_proj)], -1)
        return encoded_xy
    def forward(self, inputs): #inputs: norm, wo, rgb, kd, ks, r 
        inputs = self.PE_norm(inputs[0]) + self.PE_wo(inputs[1]) + self.PE_rgb(torch.log(rgb_to_srgb(inputs[2][None, ...])[0]+1)) + self.PE_kd(inputs[3]) + self.PE_ks(inputs[4]) + self.PE_r(inputs[5]) 
        outputs = self.mlp(inputs)
        return outputs

#----------------------------------------------------------------------------
# Relighting function. (A simple implententation without blender)
#----------------------------------------------------------------------------

def sphere_2D(res):
    gy, gx = torch.meshgrid(torch.linspace( 1.0 - 1.0 / res[0], -1.0 + 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    gz = torch.ones(res[0], res[0], device='cuda')
    gz = gz - gx*gx - gy*gy
    mask = gz < 0
    img = torch.cat([gx.clone()[..., None], gy.clone()[..., None], gz.clone()[..., None]], -1)
    img[mask] = 0.
    return img
    
def sphere_2D_visual(res):
    gy, gx = torch.meshgrid(torch.linspace( 1.0 - 1.0 / res[0], -1.0 + 1.0 / res[0], res[0], device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                            indexing='ij')
    gz = torch.ones(res[0], res[0], device='cuda')
    gz = gz - gx*gx - gy*gy
    mask = gz < 0
    img = torch.cat([gx.clone()[..., None]/2+0.5, gy.clone()[..., None]/2+0.5, gz.clone()[..., None]], -1)
    img[mask] = 0.
    save_image('sphere_2D_visual.png', img.detach().cpu().numpy()) 
    return img
    
def sphere2envmap_visual(gt, img, mask, idx):
    env_map = torch.from_numpy(gt).cuda()
    res = env_map.shape
    env_map = torch.roll(env_map, int(-res[1]/2), 1)
    env_map = torch.flip(env_map, (1,))
    shape = env_map.shape
    env_map = env_map.reshape(shape[0]*shape[1], shape[2])
    env_map[idx] = img[mask]
    env_map = env_map.reshape(shape[0], shape[1], shape[2])
    save_image('sphere2envmap_visual.png', env_map.detach().cpu().numpy()) 
    return env_map
    
def relighting(env_map, img, mask, idx):
    env_map = torch.from_numpy(env_map).cuda()
    res = env_map.shape
    env_map = torch.roll(env_map, int(-res[1]/2), 1)
    env_map = torch.flip(env_map, (1,))
    shape = env_map.shape
    env_map = env_map.reshape(shape[0]*shape[1], shape[2])
    img[mask] = env_map[idx]
    return img
    
#----------------------------------------------------------------------------
# Loaders function.
#----------------------------------------------------------------------------

def load_PE_parameters(MLP, file_name):
    parameters = torch.load(file_name)
    MLP.B0 = parameters[..., 0:3]
    MLP.B1 = parameters[..., 3:6]
    MLP.B2 = parameters[..., 6:9]
    MLP.B3 = parameters[..., 9:12]
    MLP.B4 = parameters[..., 12:15]
    MLP.B5 = parameters[..., 15:16]
    
def load_sampled_pixels(sampled_pixels, file_name):
    pixel_info = torch.load(file_name)
    info_type = ['noraml', 'w_o', 'color', 'k_d', 'k_s', 'roughness']
    for i in range(len(info_type)):
        sampled_pixels.append(pixel_info[..., 3*i:3*(i+1)])
        print('pixel '+info_type[i]+':', pixel_info[..., 3*i:3*(i+1)].shape)
    
#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

print('Loading MLP weights.')        
MLP = LightDistill()
MLP.load_state_dict(torch.load('mlp_weights.pt'))
load_PE_parameters(MLP, 'PE_parameters.pt')

print('Loading sampled pixel informations.')
sampled_pixels = []
load_sampled_pixels(sampled_pixels, 'sampled_pixels.pt')

with torch.no_grad():
    print('Predicting color of sampled probe through LightDistill MLP.')
    start_time = time.time()
    rgb = MLP(sampled_pixels)
    print('Calculating distribution of sampled probe & Stacking probes for environment map.')
    norm = sampled_pixels[0]
    wo = sampled_pixels[1]
    r = sampled_pixels[5]
    size = 4096
    split = int(rgb.shape[0] / size)
    for i in range(split+1):  # split the stacking process to prevent CUDA out of memory
        if i == 0:
            rgb_sum, weight_sum = distribution(norm[0:size], wo[0:size], r[0:size], rgb[0:size])
        else:
            rgb_split, weight_split = distribution(norm[size*i:size*(i+1)], wo[size*i:size*(i+1)], r[size*i:size*(i+1)], rgb[size*i:size*(i+1)])
            rgb_sum += rgb_split
            weight_sum += weight_split
    rgb_sum[(weight_sum!=0).reshape(weight_sum.shape[0])] = rgb_sum[(weight_sum!=0).reshape(weight_sum.shape[0])] / weight_sum[(weight_sum!=0).reshape(weight_sum.shape[0])]
    env_map = torch.flip(rgb_sum, dims=(0,)).reshape(emap_res[0], emap_res[1], 3) #flip the env map to fit the coordinate setting of ALP.
    env_map = torch.permute(resize(torch.permute(env_map, (2, 0, 1))), (1, 2, 0)) #resize from [128, 256] to [512, 1024] to match ALP resolution
    end_time = time.time()
    #print('Duration:', round(end_time - start_time, 2), 'seconds.')
    print('done')
    
print('Saving images.')
ours = env_map.detach().cpu().numpy()
save_image('LightDistill_env.png', ours) 
gt = load_image('gt_env.png')
alp = load_image('ALP_env.png')
nvdiffrec = load_image('Nvdiffrec_env.png')
comparison = np.concatenate([gt, alp, nvdiffrec, ours], 1)
save_image('Comparison_env.png', comparison)

#Relighting 
print('Relighting on the mirror material ball.')
img = sphere_2D([512, 512])
visual = sphere_2D_visual([512, 512])
mask = img[..., 2] > 0.
idx = cart_to_equil(img, [512, 1024])
idx = (idx[mask][..., 0]*1024 + idx[mask][..., 1]).long()
visual_map = sphere2envmap_visual(gt, visual, mask, idx)

env_map = [gt, alp, nvdiffrec, ours]
methods = ['gt', 'ALP', 'Nvdiffrec', 'LightDistill']
comparison = []
for i in range(len(methods)):
    relighted = relighting(env_map[i], img, mask, idx)
    comparison.append(relighted.clone())
    save_image(methods[i]+'_relit.png', relighted.detach().cpu().numpy()) 
save_image('combined_visual.png', torch.cat([visual, visual_map, comparison[0]], 1).detach().cpu().numpy())
comparison = torch.cat(comparison, 1)
save_image('Comparison_relit.png', comparison.detach().cpu().numpy()) 
print('The relighted results in this demo are produced by a simple implementation for mirror sphere without Blender, which are not as precise as results preduced by Blender.')