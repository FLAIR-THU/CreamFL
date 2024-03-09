import os, gzip, shutil
import torch
import torchvision
import numpy as np
import random, math
import pandas
import csv
from concurrent.futures.thread import ThreadPoolExecutor


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
 
class Helper:
  #All directories are end with /
  
  @staticmethod
  def get_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
  
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
  
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
  
  @staticmethod
  def pairwise_L2(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

  @staticmethod
  def network_norm(Module):
      norm = 0.0
      counter = 0.0
      for name, param in Module.named_parameters():
          if 'weight' in name:
              counter += 1
              norm += param.cpu().clone().detach().norm()/torch.sum(torch.ones(param.shape))
          elif 'bias' in name:
              counter += 1
              norm += param.cpu().clone().detach().norm()/torch.sum(torch.ones(param.shape))
      return (norm/counter).item()
   
  ###======================== Systems ======================== ####
  @staticmethod
  def multithread(max_workers, func, *args):  
      with ThreadPoolExecutor(max_workers=20) as executor:
          func(args)
          
  ###======================== Utilities ====================== ####
  @staticmethod
  def add_common_used_parser(parser):
    #=== directories ===
    parser.add_argument('--exp_name', type=str, default='Test', help='The name for different experimental runs.')
    parser.add_argument('--exp_dir', type=str, default='../../experiments/', help='Locations to save different experimental runs.')
    
    #== plot figures ===
    parser.add_argument('--plot', dest='plot', action='store_true')
    parser.set_defaults(plot=False)
    return parser
     
  @staticmethod
  def get_save_dirs(exp_dir, exp_name):
    exp_dir = os.path.join(exp_dir, exp_name)
    save_dirs = dict()
    save_dirs['codes']  = os.path.join(exp_dir, 'codes/')
    save_dirs['checkpoints']  = os.path.join(exp_dir, 'checkpoints/')
    save_dirs['logs']  = os.path.join(exp_dir, 'logs/')
    save_dirs['figures']  = os.path.join(exp_dir, 'figures/')
    save_dirs['results']  = os.path.join(exp_dir, 'results/')
    for name, _dir in save_dirs.items():
      if not os.path.isdir(_dir):
        print('Create {} directory: {}'.format(name, _dir))
        os.makedirs(_dir)
    return save_dirs

  @staticmethod  
  def backup_codes(src_d, tgt_d, save_types=['.py', '.txt', '.sh', '.out']):
    for root, dirs, files in os.walk(src_d):
      for filename in files:
        type_list = [filename.endswith(tp) for tp in save_types]
        if sum(type_list):
          file_path = os.path.join(root, filename)
          tgt_dir   = root.replace(src_d, tgt_d)
          if not os.path.isdir(tgt_dir):
            os.makedirs(tgt_dir)
          shutil.copyfile(os.path.join(root, filename), os.path.join(tgt_dir, filename))
      
  @staticmethod
  def try_make_dir(d):
    if not os.path.isdir(d):
      # os.mkdir(d)
      os.makedirs(d) # nested is allowed

  @staticmethod
  def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s
  
  @staticmethod
  def set_seed(seed):
    if seed == 0:
      print("Random seed is used, cudnn.deterministic is set to False.")
      torch.backends.cudnn.deterministic = False
      torch.backends.cudnn.benchmark = True
      return

    print(f"Seed {seed} is used, cudnn.deterministic is set to True.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    #To let the cuDNN use the same convolution every time
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

  ###======================== Logs ======================== ####
  @staticmethod
  def log(logf, msg, mode='a', console_print=True):
    with open(logf, mode) as f:
        f.write(msg + '\n')
    if console_print:
        print(msg)
     
     
  @staticmethod
  def write_dict2csv(log_dir, write_dict, mode="a"):
    for key in write_dict.keys():
      with open(log_dir + key + '.csv', mode) as f:
        if isinstance(write_dict[key], str):
          f.write(write_dict[key])
        elif isinstance(write_dict[key], list):
          writer = csv.writer(f)
          writer.writerow(write_dict[key])
        else:
          raise ValueError("write_dict has wrong type")
  
  
   ###======================== Visualization ================= ###
  @staticmethod
  def save_images(samples, sample_dir, sample_name, offset=0, nrows=0):
    if nrows == 0:
      bs = samples.shape[0]
      nrows = int(bs**.5)
    if offset > 0:
      sample_name += '_' + str(offset)
    save_path = os.path.join(sample_dir, sample_name + '.png')
    torchvision.utils.save_image(samples.cpu(), save_path, nrow=nrows, normalize=True) 


  ###========== Plot publication quality figures ============ ###
  @staticmethod
  def plot_figure(csv_path_list,
                  plot_fn,
                  read_fn,
                  num_axis=(1, 1), 
                  legenddict = {'bbox':(0.253, 1.237), 'ncol':1},
                  pad = 0.0,
                  width = 3.487,
                  height = 3.487 / 1.618,
                  text_list = [],
                  save_path = './plot.pdf',
                  bbox_inches = 'tight',
                  ):
      '''
      Possible pacakages for the function to work:
      
      python     3.7.10
      matplotlib 3.4.2
  
      !sudo apt install texlive-fonts-recommended texlive-fonts-extra
      !sudo apt install dvipng
      !sudo apt install texlive-full
      !sudo apt install texmaker
      !sudo apt install ghostscript
      
      Parameters:
      --------------
      text_list = [{'x':-6.55,'y': -12.0, 'text':'(a) FID($\downarrow$)', 'fontsize': 16},
                   {'x':-2.945,'y': -12.0, 'text':'(b) class-FID($\downarrow$)', 'fontsize': 16},
                  ]
                  
      bbox_inches = Bbox(np.array([[-0.22, -0.7], [12.9, 1.93]]))
      
      read_fn 
      function used to read a list of csv files
      returns a list of dictionaries to be processed by plot_fn
      
      plot_fn
      function used to plot figures given a dictionary of hyperparameters and an axis
      no returns
      
      
      subplots
      [0,0]             [0,1]  ... [0, num_axis[1]]
      [1,0]
      .
      .
      .
      [num_axis[0], 0]         ...
      --------------
      '''
      import matplotlib
      import matplotlib.pyplot as plt
      from scipy.interpolate import make_interp_spline
      import matplotlib.font_manager as font_manager
      import matplotlib
      from matplotlib.transforms import Bbox
      import numpy as np
      import pandas
      
      #============ main part of the plot figure code ===============
      matplotlib.style.use('seaborn')
    
      plt.rc('font', family='serif', serif='Times')
      plt.rc('text', usetex=True)
      plt.rc('xtick', labelsize=12)
      plt.rc('ytick', labelsize=12)
      plt.rc('axes', labelsize=12)  
      
      fig, ax = plt.subplots(num_axis[0], num_axis[1])
      plt.show(block=False)
      
      if min(num_axis[0], num_axis[1]) > 1:
          ax = [ax[i,j] for i in range(num_axis[0]) for j in range(num_axis[1])]

      plots_dict_list = read_fn(csv_path_list)
      for i in range(len(ax)):
          plot_fn(ax = ax[i], plots_dict = plots_dict_list[i])
      
      # === global legends ===
      #handles, labels = ax[0].get_legend_handles_labels()
      #fig.legend(handles, labels, bbox_to_anchor=legenddict['bbox'], ncol=legenddict['ncol'], loc='center', fontsize=16)    
      #fig.tight_layout(pad=pad)
          
      for text in text_list:
        plt.text(text['x'], text['y'], text['text'], fontsize=text['fontsize'])
      
      fig.set_size_inches(width, height)
      fig.tight_layout()
      fig.savefig(save_path, bbox_inches=bbox_inches)
  
  @staticmethod
  def normal_read_fn(csv_path_list):
      plots_dict_list = []
      for i in range(len(csv_path_list)):
          plots_dict = {}
          
          path = csv_path_list[i]
          raw  = pandas.read_csv(path, header=0)
          df   = pandas.DataFrame(raw)
           
          plots_dict['names']       = list(df.columns)[1:]
          plots_dict['df']          = [pandas.DataFrame(df[name]) for name in plots_dict['names']]
          for d in plots_dict['df']: d.columns = ['y']
          plots_dict['xlabel']      = None
          plots_dict['ylabel']      = None 
          plots_dict['y_lim']       = None 
          plots_dict['yticks']      = None 
          plots_dict['xticks']      = None
          plots_dict['yticklabels'] = None
          plots_dict['xticklabels'] = None
          plots_dict['alpha']       = 0.25
          plots_dict['smooth']      = False
          plots_dict['sigma']       = False
          plots_dict['markon']      = False
          
          plots_dict_list.append(plots_dict) 
      #=== special settings for each plots === 
      plots_dict_list[1]['names'] = [name.replace('_', '\_') for name in plots_dict_list[1]['names']]
      plots_dict_list[2]['names'] = [name.replace('_', '\_') for name in plots_dict_list[2]['names']]
      return plots_dict_list

  @staticmethod
  def normal_plot_fn(ax, plots_dict):
    df_list     = plots_dict['df']           #list
    full_names  = plots_dict['names']        #list
    xlabel      = plots_dict['xlabel']       #str
    ylabel      = plots_dict['ylabel']       #str
    y_lim       = plots_dict['y_lim']        #list
    yticks      = plots_dict['yticks']       #list
    xticks      = plots_dict['xticks']       #list
    yticklabels = plots_dict['yticklabels']  #list
    xticklabels = plots_dict['xticklabels']  #list
    alpha       = plots_dict['alpha']
    
    c = ['darkblue', 'darkgreen', 'darkorange', 'darkred', 'darkslategray', 'darkmagenta', 'gold']
    m = ['s', 'o', '^', '*', "X", 'D']
    
    for i in range(len(df_list)):
      y = df_list[i]['y'].to_numpy()[1:]
      x = np.arange(len(y))
      markon = x if len(x) < 10 else x[::len(x)//10]
      if plots_dict['smooth']:
        x_smooth = np.linspace(x.min(), x.max(), 100) 
        spl = make_interp_spline(x, y, k=2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, label=full_names[i], color=c[i], marker=m[i] if plots_dict['markon'] else None, markevery=markon)
      else:
        ax.plot(x, y, marker=m[i] if plots_dict['markon'] else None, label=full_names[i], color=c[i], markevery=markon)
      
      if plots_dict['sigma']:
        mu    = df_list[i]['mean'].to_numpy()
        sigma = df_list[i]['std'].to_numpy()
        if i == 2:   
          ax.fill_between(x, mu+sigma, mu-sigma, facecolor=c[i], alpha=alpha*2)
        else:
          ax.fill_between(x, mu+sigma, mu-sigma, facecolor=c[i], alpha=alpha)
  
    # Add some text for labels, title and custom x-axis tick labels, etc.
    if xlabel      is not None: ax.set_xlabel(xlabel, size=12)
    if ylabel      is not None: ax.set_ylabel(ylabel,size=12)
    if y_lim       is not None: ax.set_ylim(y_lim)
    if xticks      is not None: ax.set_xticks(xticks)
    if yticks      is not None: ax.set_yticks(yticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    ax.legend(loc='upper left')
    ax.grid(True)

      