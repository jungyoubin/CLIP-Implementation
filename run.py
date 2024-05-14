import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from clip import ClipModel
from dataset import ClipDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, required=True) # 이미지 경로
    parser.add_argument('--annotations_file', type=str, required=True) # Annotations JSON 파일 경로
    parser.add_argument('--model_weights_path', type=str, required=True) # model weight 경로
    parser.add_argument('--image_path', type=str, required=True) # predict 이미지 경로
    parser.add_argument('--label_list', nargs='+', required=True) # zero-shot prediction label
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    return parser.parse_args()


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

transform = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


######################################################
##################### Train ##########################
######################################################
def train_model(model, dataloader, optimizer, num_epochs=10):
    model = model.to(device)
    save_path = './model'
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i, data in progress_bar:
            images, caption = data['image'], data['caption']
            images = images.to(device)
            labels = torch.arange(images.size(0)).to(device)
            inputs = tokenizer(caption, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

            input_ids = inputs['input_ids'].squeeze().to(device)
            attention_mask = inputs['attention_mask'].squeeze().to(device)

            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.T, labels)
            loss = (loss_i + loss_t) / 2
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (i + 1)})
        print(f'Epoch {epoch + 1}/{num_epochs}, Avg Loss: {running_loss / len(dataloader):.4f}')
        epoch_save_path = f'./model/{save_path}_epoch_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_save_path)

        print(f'Model saved to {epoch_save_path}')

    print('Training completed')
    return model


######################################################
################## Prediction ########################
######################################################
def zeroshot_prediction(model, image_path, label_list, image_transform, tokenizer):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(device)

    text_inputs = ["A photo of a {}.".format(label) for label in label_list]
    inputs = tokenizer(text_inputs, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    input_ids = inputs['input_ids'].squeeze().to(device)
    attention_mask = inputs['attention_mask'].squeeze().to(device)

    with torch.no_grad():
        logits = model(image, input_ids, attention_mask)

        probabilities = F.softmax(logits, dim=-1)
        top_pred_index = probabilities.argmax().item()
    return label_list[top_pred_index], probabilities


if __name__ == "__main__":
    args = parse_args()

    dataset = ClipDataset(args.img_dir, args.annotations_file, tokenizer, transform, device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    image_embedding_dim = 2048
    text_embedding_dim = 768
    model = ClipModel(image_embedding_dim, text_embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trained_model = train_model(model, dataloader, optimizer, args.num_epochs)

    model_weights = torch.load(args.model_weights_path)
    model.load_state_dict(model_weights['model_state_dict'])
    model.to(device)
    model.eval()

    # Zero-shot prediction
    predicted_label, probabilities = zeroshot_prediction(model, args.image_path, args.label_list, transform, tokenizer)
    print(f"Predicted label: {predicted_label}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")