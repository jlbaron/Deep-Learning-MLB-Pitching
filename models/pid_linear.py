import torch
import torch.nn as nn

class PitchIdentifierLinear(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.1, features=15, num_classes=16, device='cpu'):
        super(PitchIdentifierLinear, self).__init__()
        self.features = features

        self.feature_min = torch.tensor([-10.5433282251965, -5.18366373737265, 33.9, 32.4, 1.214, -0.002, -59.29009,
                                         -0.524374801782275, -77.1714212303812, -24.869, -153.362, -19.7706017189303, -90.0, 0.1, 23.3], device=device)
        self.feature_max = torch.tensor([12.9529095060724, 14.8862417199696, 105.0, 96.9, 6539.259, 360.001, 40.978, 54.057,
                                         22.3052008380727, 25.15, -62.7413771784932, 27.815,  269.4, 224889.3, 36.4], device=device)

        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(features, hidden_dim, device=device)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, num_classes, device=device)
        
    def normalize_features(self, features, feature_num=15):
        '''
        Feature: px, Max Value: 12.9529095060724, Min Value: -10.5433282251965
        Feature: pz, Max Value: 14.8862417199696, Min Value: -5.18366373737265
        Feature: start_speed, Max Value: 105.0, Min Value: 33.9
        Feature: end_speed, Max Value: 96.9, Min Value: 32.4
        Feature: spin_rate, Max Value: 6539.259, Min Value: 1.214
        Feature: spin_dir, Max Value: 360.001, Min Value: -0.002
        Feature: ax, Max Value: 40.978, Min Value: -59.29009
        Feature: ay, Max Value: 54.057, Min Value: -0.524374801782275
        Feature: az, Max Value: 22.3052008380727, Min Value: -77.1714212303812
        Feature: vx0, Max Value: 25.15, Min Value: -24.869
        Feature: vy0, Max Value: -62.7413771784932, Min Value: -153.362
        Feature: vz0, Max Value: 27.815, Min Value: -19.7706017189303
        Feature: break_angle, Max Value: 269.4, Min Value: -90.0
        Feature: break_length, Max Value: 224889.3, Min Value: 0.1
        Feature: break_y, Max Value: 36.4, Min Value: 23.3
        '''
        normalized_features =  (features.long() - self.feature_min) / (self.feature_max - self.feature_min)
        normalized_features = torch.FloatTensor(normalized_features)
        return normalized_features
    def forward(self, inputs):
        x = self.normalize_features(inputs, feature_num=self.features)

        x = self.drop(self.fc1(x))
        x = self.gelu(x)
        logits = self.fc2(x) #pass through classifier for class scores
        return logits
