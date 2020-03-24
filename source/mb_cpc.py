import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.notebook import tqdm
from .bert_pretrain_convert.state_dict_translate import translate_from_hugginface_to_torch_BERT
from .sound_modules import SoundCNNEncoder
from .mb_base import MusicBert
from .utils import torch_count_nonzero, generator_repeat, Timer


class MusicBertContrastivePredictiveCoding(nn.Module):

    def __init__(self, dim_sound, n_convs=4, num_tokens=30522, name="music_bert_masked_lm"):
        super(MusicBertMaskedLM, self).__init__()
        
        self.BERT = MusicBert(dim_sound = dim_sound,
                              num_tokens = num_tokens,
                              name = name,
                              n_convs=n_convs)
        
        self.PATH = "models/" + name + ".pth"
        
        self.out_sound = nn.Linear(self.BERT.d_model, dim_sound)
        self.out_tokens= nn.Linear(self.BERT.d_model, num_tokens)
        self.out_cls   = nn.Linear(self.BERT.d_model, 1)
        
        ## Optimizer parameters
        
        lr = 1e-3 # learning rate
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        
        self.AE_loss_curve = []
        self.loss_curve = []
        self.validation_curve = []
        self.cls_loss_curve = []
        self.tokens_loss_curve = []
        self.sound_loss_curve = []
        
        self.params = self.parameters()
        
        
    def get_device(self):
        return next(self.parameters()).device
    
    
    def load_pretrained(self):
        self.BERT.load_pretrained()
        
        
    def save_model(self):
#         save_obj = {
#             'epoch': epoch,
#             'model_state_dict': self.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss
#             }
        torch.save(self.state_dict(), self.PATH)
    
    
    def load_model(self):
        self.load_state_dict(torch.load(self.PATH))
        
        
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    def forward(self, src_sound=None, src_tokens=None):
        
        tokens_len = len(src_tokens)
        entangled_sequence = self.BERT(src_sound, src_tokens)
        
        # This is already discarding the information from SEP and CLS tokens
        processed_tokens = entangled_sequence[1:tokens_len+1]
        processed_sound = entangled_sequence[tokens_len+2:]
        cls_token = entangled_sequence[0] # Bert Pooling
        
        return  self.out_cls(cls_token), \
                self.out_tokens(processed_tokens), \
                self.out_sound(processed_sound)
                
    
    
    def __masking_loss(self, batch):
        
        sample_sound = batch['song_features'].to(self.get_device())
        masked_sound, sound_mask = self.mask_samples(sample_sound,
                                                     ignore_pad=False,
                                                     mask_token=0)

        sample_tokens = batch['full_text'].long().to(self.get_device())
        masked_tokens, tokens_mask = self.mask_samples(sample_tokens,
                                                       ignore_pad=True,
                                                       mask_token=1)
        
        sample_cls = batch['cls'].float().to(self.get_device())
        
        out_cls, out_tokens, out_sound = self.forward(masked_sound.permute(1,0,2),
                                                      masked_tokens.permute(1,0))

        masked_sound_gt = torch.gather(sample_sound, 1, sound_mask)
        masked_sound_out = torch.gather(out_sound.permute(1,0,2), 1, sound_mask)

        masked_tokens_gt = torch.gather(sample_tokens, 1, tokens_mask)
        masked_tokens_out = torch.gather(out_tokens.permute(1,0,2), 1,
                                 tokens_mask.unsqueeze(-1).repeat(1,1,self.BERT.num_tokens))
        
        cls_loss = self.cls_criterion(out_cls, sample_cls)
        tokens_loss = self.tokens_criterion(masked_tokens_out.permute(0,2,1), masked_tokens_gt)
        sound_loss  = self.sound_criterion(masked_sound_gt, masked_sound_out)
        
        return cls_loss, tokens_loss, sound_loss

    
    def mask_samples(self, x, ignore_pad=True, mask_token=1):
        batch_size = x.shape[0]
        sequence_length = x.shape[1]
        masking_elements = sequence_length//10
        
        if masking_elements < 1:
            masking_elements = 1
            
        mask = []
        for sample in x:
            if ignore_pad:
                sequence_length = torch_count_nonzero(sample) # 0 is the PAD element
                
            # Samples from permutations
            sample_mask = torch.rand(masking_elements, device=x.device)
            sample_mask = sample_mask*(sequence_length-1)+1
            
            # (batch, n_masks) indexex of masked vectors and also avoid CLS vector
            mask.append(sample_mask.unsqueeze(0).long())
        
        mask = torch.cat(mask, 0)
        x = x.clone()
        for i in range(batch_size):
            x[i,mask[i]] = mask_token
            
        if len(x.shape)>2:
            emb_dim = x.shape[2]
            process_mask = mask.unsqueeze(-1).repeat(1,1,emb_dim)
        else:
            process_mask = mask

        return x, process_mask
    
    
    def pretrain_model(self, pretrain_dataloader, val_dataloader, epochs, eval_per_epoch=10):

        self.train()
        pretrain_length = len(pretrain_dataloader)
        save_interval = eval_interval = pretrain_length//eval_per_epoch

        pbar = tqdm(generator_repeat(pretrain_dataloader, epochs),
                    total = pretrain_length*epochs,
                    desc="Train ({:d}/{:d} Epoch) - Loss...".format(1, epochs))
        for i, batch in enumerate(pbar):
            self.optimizer.zero_grad() # Reset Grad

            cls_loss, tokens_loss, sound_loss = self.__masking_loss(batch)
                                                         
            loss = cls_loss + tokens_loss + sound_loss # Normalize for the length disparity
            # loss = sound_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, 0.5)
            self.optimizer.step()
            
            self.loss_curve.append(loss.item())
            self.cls_loss_curve.append(cls_loss.item())
            self.tokens_loss_curve.append(tokens_loss.item())
            self.sound_loss_curve.append(sound_loss.item()) 
            
            if i%10 == 0:
                n_epoch = i//pretrain_length
                display_loss = np.mean(self.loss_curve[-100:])
                pbar.set_description('Pre-train ({:d}/{:d} Epoch) - Loss {:.6f}'\
                                     .format(n_epoch+1, epochs, display_loss))
                
            if i%eval_interval == -1%eval_interval and val_dataloader is not None:
                tqdm.write("Evaluating...                           ", end="\r")
                eval_loss = self.evaluate(val_dataloader)
                self.validation_curve.append(eval_loss)
                tqdm.write('Eval Loss ({:d} steps) {:.4f}'.format(i, eval_loss), end="\r")

            if i%save_interval == -1%save_interval:
                self.save_model()
    
    
    def pretrain_AE(self, dataloader, epochs, skip_encoder=False):
        
        self.train()
        length = len(dataloader)
        
        self.BERT.set_encoder_grad(False)
        self.BERT.set_convs_grad(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        pbar = tqdm(generator_repeat(dataloader, epochs),
                    total = length*epochs,
                    desc="AE-Train ({:d}/{:d} Epoch) - Loss...".format(1, epochs))
        for i, batch in enumerate(pbar):
            optimizer.zero_grad() # Reset Grad
        
            sample_sound = batch['song_features'].to(self.get_device())
            sample_sound = sample_sound.permute(1,0,2)

            optimizer.zero_grad()

            outputs = self.BERT(src_sound = sample_sound, add_special_tokens=False, skip_encoder=skip_encoder)
            outputs = self.out_sound(outputs)

            # compute training reconstruction loss
            train_loss = criterion(outputs, sample_sound)
            train_loss.backward()
            optimizer.step()

            self.AE_loss_curve.append(train_loss.item())
            
            if i%10 == 0:
                n_epoch = i//length
                display_loss = np.mean(self.AE_loss_curve[-100:])
                pbar.set_description('AE-train ({:d}/{:d} Epoch) - Loss {:.6f}'\
                                     .format(n_epoch+1, epochs, display_loss))
            
        self.BERT.set_encoder_grad(True)
        self.BERT.set_convs_grad(False)
        
    def evaluate(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            
            losses = []
            with Timer() as t:
                for i, batch in enumerate(val_dataloader):
                    print("{:d}%, Elapsed time {:d} s                  ".format(i*100//len(val_dataloader), int(t())), end="\r")
                    cls_loss, tokens_loss, sound_loss = self.__masking_loss(batch)
                    loss = cls_loss + tokens_loss + sound_loss # Normalize for the length disparity

                    losses.append(loss.item())
        self.train()      
        return np.mean(losses)