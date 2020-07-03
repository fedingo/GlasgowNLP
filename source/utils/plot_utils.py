from source.models_base.mb_vggish import MusicBertVGGish
from source.models_base.mb_base import MusicBertBase
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import numpy as np
import torch


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth[box_pts:-box_pts]


def visualize_vectors(BERT, dataset):
    
    index = np.random.randint(len(dataset))
    sample_sound = torch.tensor(dataset[index]['song_features']).to(BERT.get_device()).unsqueeze(1)
    
    output, extras = BERT(sample_sound, return_extras=True, add_special_tokens=False)
    
    vggish_features = extras['encoded_sound_1'].cpu().squeeze().numpy()

    plot_layers([vggish_features], "VGGish Features", graph_height= 4)
    plot_layers(extras['layers_out'], "Layer", graph_height=4.2)
    
    
    print(np.mean(np.std(vggish_features, axis = 0)))
    for i, layer_out in enumerate(extras['layers_out']):
        print(np.mean(np.std(layer_out.cpu().squeeze().numpy(), axis = 0)))
    
    
def plot_layers (matrices_list, prefix_name = "Title", graph_height=2):
    
    length = len(matrices_list)
    fig, axs = plt.subplots(length, 1, figsize=(10, graph_height*length))
    
    if length == 1:
        axs = [axs]
    
    for i, layer_out in enumerate(matrices_list):
        if type(layer_out) == torch.Tensor:
            layer_out = layer_out.cpu().squeeze().detach().numpy()[:,:128]
            
        cax = axs[i].matshow(layer_out)
        axs[i].title.set_text(prefix_name + " %d" % i)
        axs[i].title.set_position([.5, 1.15])
        fig.colorbar(cax, ax=axs[i], orientation="horizontal")
        
    # Adjust to remove space before and after the plot
    plt.subplots_adjust(top=0.99, bottom=0.01)
    
    
def plot_curve(curve, eval_per_epoch, color = "red", metric_label="Loss"):
    
    x = np.array(range(len(curve)))//eval_per_epoch
    plt.plot(x, curve, color)
    plt.ylabel(metric_label)
    plt.xlabel('Steps')
    plt.show()