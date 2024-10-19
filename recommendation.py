import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score

# Define basic constants
EMBEDDING_DIM = 50
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 0.001
TOP_K = 5  # Set k for Precision@K and Recall@K

# Load datasets
user_data = pd.read_csv('user_data.csv')
video_data = pd.read_csv('video_data_with_frames.csv')
interaction_data = pd.read_csv('interaction_data.csv')

# Step 5: Calculate the number of unique tags
# Assuming tags are stored in a column named 'tags' in video_data
all_tags = set()
for tags in video_data['tags'].dropna():  # Drop NaN if there are any
    all_tags.update(tag.strip() for tag in tags.lower().strip('#').split())

num_tags = len(all_tags)  # Number of unique tags

# Step 1: Neural Collaborative Filtering (NCF) for user-item interaction embeddings
class NCFModel(nn.Module):
    def __init__(self, num_users, num_videos, embedding_dim=EMBEDDING_DIM):
        super(NCFModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.video_embedding = nn.Embedding(num_videos, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user_id, video_id):
        user_emb = self.user_embedding(user_id)
        video_emb = self.video_embedding(video_id)
        combined = torch.cat([user_emb, video_emb], dim=-1)
        return torch.sigmoid(self.fc_layers(combined))

# Step 2: Content-based embeddings (BERT for captions, ResNet for video frames)
class ContentEmbeddings(nn.Module):
    def __init__(self):
        super(ContentEmbeddings, self).__init__()
        # BERT model for text embeddings
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # ResNet for video frame embeddings
        resnet = models.resnet50(pretrained=True)
        self.resnet_model = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer

        # Image preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def get_text_embedding(self, caption):
        inputs = self.tokenizer(caption, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token

    def get_video_embedding(self, video_frame_path):
        if not isinstance(video_frame_path, str):
            raise ValueError(f"Expected video_frame_path to be a string, but got {type(video_frame_path)}")

        try:
            # Load the image
            image = Image.open(video_frame_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}, path: {video_frame_path}")
            # Return a dummy tensor if the image fails to load
            return torch.zeros(2048)

        input_tensor = self.preprocess(image).unsqueeze(0)  # Create batch size of 1

        # Extract features using ResNet
        with torch.no_grad():  # Disable gradient tracking
            video_features = self.resnet_model(input_tensor).flatten()

        return video_features

    def forward(self, caption, video_frame_path):
        # Text embedding from BERT
        text_emb = self.get_text_embedding(caption)
        
        # Video embedding from ResNet
        video_emb = self.get_video_embedding(video_frame_path)
        
        # Ensure both embeddings have compatible shapes for concatenation
        video_emb = video_emb.expand(text_emb.size(0), -1)  # Repeat video embedding for batch size
        
        # Concatenate the embeddings
        combined = torch.cat([text_emb, video_emb], dim=-1)
        return combined

# Step 3: Hybrid model combining NCF and content embeddings
class HybridModel(nn.Module):
    def __init__(self, num_users, num_videos, num_tags, embedding_dim=EMBEDDING_DIM):
        super(HybridModel, self).__init__()
        self.ncf_model = NCFModel(num_users, num_videos, embedding_dim)
        self.tag_to_id = {}
        self.content_embeddings = ContentEmbeddings()

        # Tag embeddings
        self.tag_embedding = nn.Embedding(num_tags, embedding_dim)  # New line for tag embeddings
        
        
        # Fully connected layers for the combined model
        self.fc_combined = nn.Sequential(
            nn.Linear(2817 + embedding_dim, 128),  # NCF output + BERT text (768) + ResNet (2048)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_tag_ids(self, tags):
        """Convert tags to tag IDs and return as a tensor"""
        if isinstance(tags, list):  # If tags are a list of strings
            tag_ids = [self.tag_to_id.get(tag, 0) for tag in tags]
        elif isinstance(tags, str):  # If it's a single string
            tag_ids = [self.tag_to_id.get(tags, 0)]  # Convert to a list
        else:
            raise ValueError(f"Unexpected type for tags: {type(tags)}")

        # Convert the list of tag IDs to a tensor
        return torch.tensor(tag_ids, dtype=torch.long)

    def forward(self, user_id, video_id, caption, video_frame_path, tags):
        # NCF part
        ncf_out = self.ncf_model(user_id, video_id)

        # Content-based part
        content_emb = self.content_embeddings(caption, video_frame_path)

        # Tag embeddings
        tag_ids = self.get_tag_ids(tags)  # Implement this function to convert tags to IDs
        tag_emb = self.tag_embedding(tag_ids)  # Get tag embeddings
        

        # Reshape ncf_out to have an extra dimension for concatenation
        ncf_out = ncf_out.unsqueeze(1)  # Adds a dimension to match content_emb

        # Concatenate NCF output and content embeddings
        combined = torch.cat([ncf_out, content_emb, tag_emb], dim=-1)

        return torch.sigmoid(self.fc_combined(combined))

# Step 4: Dataset and DataLoader with interaction weighting
class RecommendationDataset(Dataset):
    def __init__(self, interaction_data, user_data, video_data):
        self.interaction_data = interaction_data
        self.user_data = user_data
        self.video_data = video_data
        
        self.tag_to_id = {tag: idx for idx, tag in enumerate(all_tags)}  # Map tags to unique IDs

        # Interaction type weights
        self.interaction_weights = {
            'view': 0.1,
            'like': 1.0,
            'comment': 0.5,
            'share': 1.5
        }
    def get_tag_ids(self, tags):
        # Convert tags to lowercase and split into a list
        tag_list = tags.lower().strip('#').split()
        tag_ids = []
        for tag in tag_list:
            # Get the ID for each tag. You may want to map tags to indices beforehand.
            tag_id = self.tag_to_id.get(tag, -1)  # Assuming you have a dictionary mapping tags to IDs
            if tag_id != -1:
                tag_ids.append(tag_id)
        
        return torch.tensor(tag_ids)
    def __len__(self):
        return len(self.interaction_data)

    def __getitem__(self, idx):
        row = self.interaction_data.iloc[idx]
        user_id = torch.tensor(row['user_id'] - 1, dtype=torch.long)  # Adjust for zero indexing
        video_id = torch.tensor(row['video_id'] - 1, dtype=torch.long)

        # Adjust interaction based on type
        interaction_weight = self.interaction_weights.get(row['interaction_type'], 0.0)
        interaction = torch.tensor(interaction_weight)

        # Get caption, video frame path, and tags
        caption = self.video_data.loc[self.video_data['video_id'] == row['video_id'], 'caption'].values
        video_frame_path = self.video_data.loc[self.video_data['video_id'] == row['video_id'], 'frame_path'].values
        tags = self.video_data.loc[self.video_data['video_id'] == row['video_id'], 'tags'].values

        # Check if any of these values are empty
        if len(caption) == 0 or len(video_frame_path) == 0 or len(tags) == 0:
            print(f"Missing data for index {idx}: {row}, caption: {caption}, frame_path: {video_frame_path}, tags: {tags}")
            return None  # Return None if any value is missing

        # Ensure video_frame_path is a string and not a tuple or list
        if isinstance(video_frame_path[0], str):  # Check the first element
            return user_id, video_id, caption[0], video_frame_path[0], tags[0], interaction  # Use first element
        else:
            raise ValueError(f"Expected video_frame_path to be a string but got {type(video_frame_path)}")


# Initialize dataset and dataloader
dataset = RecommendationDataset(interaction_data, user_data, video_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model with num_tags included
model = HybridModel(num_users=len(user_data), num_videos=len(video_data), num_tags=num_tags)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss

# Initialize lists to store predictions and targets for evaluation
ncf_preds = []
ncf_targets = []
content_preds = []
content_targets = []
hybrid_preds = []
hybrid_targets = []

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        user_ids, video_ids, captions, video_frames, tags, interactions = batch
        optimizer.zero_grad()
        
        # Store NCF and hybrid outputs for metrics calculation
        outputs = []
        for i in range(len(user_ids)):
            ncf_output = model.ncf_model(user_ids[i], video_ids[i])
            hybrid_output = model(user_ids[i], video_ids[i], captions[i], video_frames[i], tags[i])

            # Collect predictions and targets
            ncf_preds.append(ncf_output.item())
            ncf_targets.append(interactions[i].item())

            hybrid_preds.append(hybrid_output.item())
            hybrid_targets.append(interactions[i].item())

            # Collect content predictions and targets
            content_emb = model.content_embeddings(captions[i], video_frames[i])
            content_pred = content_emb.mean()  # You can adjust this logic as needed
            content_preds.append(content_pred.item())
            content_targets.append(interactions[i].item())
            
            outputs.append(hybrid_output.unsqueeze(0))  # Add batch dimension

        # Concatenate the results from each iteration
        if outputs:  # Check if outputs list is not empty
            outputs = torch.cat(outputs).squeeze()  # Concatenate only if not empty
            loss = loss_fn(outputs, interactions.float().clamp(0, 1))  # Clamp to ensure values are in [0, 1]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Update learning rate
    scheduler.step()

    # Additional metrics calculations...

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss / len(dataloader):.4f}, "f"Average NCF Prediction: {np.mean(ncf_preds):.4f}, "f"Average Hybrid Prediction: {np.mean(hybrid_preds):.4f}, "f"Average Content Prediction: {np.mean(content_preds):.4f}")

# Optionally save the model
torch.save(model.state_dict(), 'hybrid_model.pth')

# After training the model
recommended_videos = []

# Collect and format the recommendations
for user_id in range(len(user_data)):
    user_recommendations = []
    for video_id in range(len(video_data)):
        caption = video_data.loc[video_data['video_id'] == video_id + 1, 'caption'].values[0]
        tags = video_data.loc[video_data['video_id'] == video_id + 1, 'tags'].values[0]
        ncf_output = model.ncf_model(torch.tensor(user_id), torch.tensor(video_id)).item()
        hybrid_output = model(torch.tensor(user_id), torch.tensor(video_id), caption, video_data.loc[video_data['video_id'] == video_id + 1, 'frame_path'].values[0], tags).item()
        user_recommendations.append((video_id + 1, caption, hybrid_output))  # Store (Reel ID, Title/Caption, Reel Score)

    # Sort the recommendations by score in descending order
    user_recommendations.sort(key=lambda x: x[2], reverse=True)
    
    # Take top K recommendations
    top_recommendations = user_recommendations[:TOP_K]
    
    # Format the output for the user
    recommended_videos.append(f"Recommended Reels for User {user_id + 1}:\n")
    recommended_videos.append(f"{'Reel ID':<10} | {'Title/Caption of Reel':<40} | {'Reel Score':<10}\n")
    recommended_videos.append("-" * 80 + "\n")
    
    for reel_id, title, score in top_recommendations:
        recommended_videos.append(f"{reel_id:<10} | {title:<40} | {score:.4f}\n")

# Print all recommended videos
print("".join(recommended_videos))
