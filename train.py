# train.py
import torch
import argparse
from tqdm import tqdm
from segment_anything import sam_model_registry

from data_loader import GuidedPromptDataset
from utils import CombinedLoss

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Setup Model
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    
    # Freeze image encoder
    for name, param in sam.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False
        elif 'mask_decoder' in name:
            param.requires_grad = True
        elif 'prompt_encoder' in name:
            param.requires_grad = False
        print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    # # 2. Setup Optimizer and Loss
    # optimizer = torch.optim.AdamW(
    #     filter(lambda p: p.requires_grad, sam.parameters()),
    #     lr=args.lr
    # )
    # loss_fn = CombinedLoss()

    # # 3. Setup Dataloader
    # dataset = GuidedPromptDataset(index_file='master_index.json', sam_model=sam)
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # # 4. Training Loop
    # for epoch in range(args.epochs):
    #     sam.train()
    #     epoch_loss = 0
        
    #     for batch in tqdm(train_loader):
    #         images = batch['image'] # [B, 3, 1024, 1024]
    #         gt_masks = batch['gt_masks'] # List of [Num_Chars, 256, 256]
    #         gt_points = batch['gt_points'] # List of [Num_Chars, 2]

    #         optimizer.zero_grad()

    #         with torch.no_grad():
    #             image_embeddings = sam.image_encoder(images)

    #         batch_loss = 0
    #         num_chars_in_batch = 0
            
    #         # Loop through each image in the batch
    #         for i in range(len(images)):
    #             points_i = gt_points[i]
    #             masks_i = gt_masks[i]
                
    #             # Loop through each character (prompt) in the image
    #             for j in range(len(points_i)):
    #                 prompt_point = points_i[j].unsqueeze(0).unsqueeze(0)
    #                 prompt_label = torch.tensor([[[1]]], device=device) # Foreground point
    #                 target_mask = masks_i[j].unsqueeze(0).unsqueeze(0) # [1, 1, 256, 256]
    #                 num_chars_in_batch += 1

    #                 sparse_emb, dense_emb = sam.prompt_encoder(
    #                     points=(prompt_point, prompt_label), boxes=None, masks=None
    #                 )

    #                 pred_masks, pred_iou = sam.mask_decoder(
    #                     image_embeddings=image_embeddings[i].unsqueeze(0),
    #                     image_pe=sam.prompt_encoder.get_dense_pe(),
    #                     sparse_prompt_embeddings=sparse_emb,
    #                     dense_prompt_embeddings=dense_emb,
    #                     multimask_output=True,
    #                 ) # [1, 3, 256, 256]

    #                 # Find the best of the 3 predicted masks
    #                 # Note: You could also use pred_iou here, but direct dice is a strong signal
    #                 dice_scores = [dice_loss(m, target_mask) for m in pred_masks.squeeze(0)]
    #                 best_mask_idx = torch.argmin(torch.tensor(dice_scores))
    #                 best_pred_mask = pred_masks[:, best_mask_idx, :, :].unsqueeze(1)
                    
    #                 loss = loss_fn(best_pred_mask, target_mask)
    #                 batch_loss += loss

    #         if num_chars_in_batch > 0:
    #             avg_loss = batch_loss / num_chars_in_batch
    #             avg_loss.backward()
    #             optimizer.step()
    #             epoch_loss += avg_loss.item()
        
    #     print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader)}")
        
    #     # Save model checkpoint
    #     torch.save(sam.state_dict(), f"finetuned_models/finetuned_sam_epoch_{epoch+1}.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='vit_b')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1) # Start with 1 due to memory
    args = parser.parse_args()
    main(args)